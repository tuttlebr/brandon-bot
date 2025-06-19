from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from openai import OpenAI
from pymilvus import MilvusClient
from utils.environment import *


@dataclass
class SearchConfig:
    """Configuration for similarity search parameters"""

    collection_name: str
    uri: str
    db_name: str
    vector_field: str = "embedding"
    radius: float = None
    range_filter: float = None
    topk: int = None
    output_fields: List[str] = None

    def __post_init__(self):
        if self.output_fields is None:
            self.output_fields = [
                "reference_id",
                "title",
                "source",
                "text",
            ]


class EmbeddingCreator:
    """Handles creation of embeddings using OpenAI API"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.model = model
        self.embed_model = OpenAI(api_key=api_key, base_url=base_url)

    def create_passage(self, input_text: str) -> Dict[str, Any]:
        """Create embedding for a passage"""
        return self.embed_model.embeddings.create(
            input=input_text, model=self.model, extra_body={"input_type": "passage", "truncate": "END"},
        )

    def create_query(self, input_text: str) -> Dict[str, Any]:
        """Create embedding for a query"""
        return self.embed_model.embeddings.create(
            input=input_text,
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )

    def create_formatted_query(self, input_text: str) -> List[float]:
        """Create and format query embedding"""
        embedding_info = self.create_query(input_text)
        return [embedding_info.data[0].embedding]


class SimilaritySearch:
    """Handles vector similarity search operations"""

    def __init__(
        self,
        collection_name: str,
        uri: str,
        db_name: str,
        vector_field: str = "embedding",
        radius: float = 1.25,
        range_filter: float = 0.001,
        output_fields: List[str] = None,
        topk: int = 15,
    ):
        self.config = SearchConfig(
            collection_name=collection_name,
            uri=uri,
            db_name=db_name,
            vector_field=vector_field,
            radius=radius,
            range_filter=range_filter,
            output_fields=output_fields,
            topk=topk,
        )
        self.search_params = self._initialize_search_params()
        self.client = self._initialize_client()

    def _initialize_search_params(self) -> Dict[str, Any]:
        """Initialize search parameters"""
        return {
            "metric_type": "L2",
            "index_type": "FLAT",
            "params": {"radius": self.config.radius, "range_filter": self.config.range_filter,},
        }

    def _initialize_client(self) -> MilvusClient:
        """Initialize and configure Milvus client"""
        client = MilvusClient(
            collection_name=self.config.collection_name,
            uri=self.config.uri,
            vector_field=self.config.vector_field,
            db_name=self.config.db_name,
            overwrite=False,
        )
        client.load_collection(collection_name=self.config.collection_name)
        return client

    def search(self, data: List[float]) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        results = self.client.search(
            data=data,
            limit=self.config.topk,
            collection_name=self.config.collection_name,
            search_params=self.search_params,
            output_fields=self.config.output_fields,
        )
        logging.debug("Vector search results: %s", results[0])

        return results

    def reranker(self, query: str, embedding_response: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Rerank search results using external reranker service"""
        from utils.environment import NVIDIA_API_KEY, RERANKER_ENDPOINT, RERANKER_MODEL

        payload = self._prepare_reranker_payload(query, embedding_response, RERANKER_MODEL)

        if not payload["passages"]:
            return None

        try:
            reranker_response = self._call_reranker_service(payload, RERANKER_ENDPOINT, NVIDIA_API_KEY)
            combined_results = self._combine_results(embedding_response, reranker_response)
            validated_results, stats = self._remove_outliers(combined_results[0], std_threshold=1.0, key="logit")
            limit = int(self.config.topk * 0.5)
            return [validated_results[:limit]]
        except Exception as e:
            logging.error("Reranking failed: %s", str(e))
            return None

    def _prepare_reranker_payload(
        self, query: str, embedding_response: List[Dict[str, Any]], model: str
    ) -> Dict[str, Any]:
        """Prepare payload for reranker service"""
        return {
            "model": model,
            "query": {"text": query},
            "passages": [
                {"text": result["entity"]["text"]} for result in embedding_response[0] if result["entity"]["text"]
            ],
            "truncate": "END",
        }

    def _call_reranker_service(self, payload: Dict[str, Any], endpoint: str, api_key: str) -> Dict[str, Any]:
        """Call reranker service with prepared payload"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "tapplication/json",
        }

        logging.debug(f"Payload: {payload}")
        with requests.Session() as session:
            response = session.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            logging.info('HTTP Request: POST %s "HTTP/1.1 200 OK"', endpoint)
            return response.json()

    def _combine_results(
        self, embedding_response: List[Dict[str, Any]], reranker_response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Combine embedding and reranker results"""
        for rank in reranker_response["rankings"]:
            embedding_response[0][rank["index"]].update({"logit": rank["logit"], "index": rank["index"]})

        rerankable_results = [result for result in embedding_response[0] if "logit" in result]

        return [rerankable_results]

    def _remove_outliers(
        self, data: Union[List[float], List[Dict[str, Any]]], std_threshold: float = 1.0, key: Optional[str] = None,
    ) -> Tuple[Union[List[float], List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Removes outliers from either a list of scores or a list of dictionaries containing scores
        using a standard deviation approach. Values greater than (mean + std_threshold * std_dev)
        will be considered outliers and removed.

        Parameters:
        data (Union[List[float], List[Dict[str, Any]]]): Either a list of similarity scores or a list of dictionaries with scores
        std_threshold (float): Number of standard deviations from the mean to use as threshold
        key (str, optional): If data is a list of dictionaries, the key for the score values

        Returns:
        Union[List[float], List[Dict[str, Any]]]: Filtered list with outliers removed
        Dict[str, Any]: Statistics about the filtering process

        Raises:
        ValueError: If data is empty or contains non-numeric values
        """
        # Input validation
        if not data:
            logging.error("Input data cannot be empty")
            return data, {"error": "Input data is empty"}

        # Debug logging
        logging.debug(f"_remove_outliers called with {len(data)} items, key='{key}', std_threshold={std_threshold}")
        if data and len(data) > 0:
            logging.debug(f"First item type: {type(data[0])}, First item: {data[0]}")

        try:
            # Extract scores if data is a list of dictionaries
            if key is not None and isinstance(data[0], dict):
                try:
                    # Extract values and validate they are numeric, keeping track of valid items
                    raw_scores = []
                    valid_items = []

                    for item in data:
                        if key not in item:
                            logging.error(f"Key '{key}' not found in item: {item}")
                            return data, {"error": f"Key '{key}' not found in data dictionary"}

                        value = item[key]
                        # Check if value is numeric (int, float) and not None/NaN
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            logging.warning(f"Skipping non-numeric value: {value}")
                            continue

                        if not isinstance(value, (int, float)):
                            logging.debug(f"Skipping non-numeric value of type {type(value)}: {value}")
                            continue

                        raw_scores.append(float(value))
                        valid_items.append(item)

                    if not raw_scores:
                        logging.error("No valid numeric values found in data")
                        return data, {"error": "No valid numeric values found"}

                    scores = np.array(raw_scores, dtype=np.float64)
                    original_data = valid_items  # Use only the items with valid scores

                except KeyError as e:
                    logging.error(f"Key '{key}' not found in data dictionary: {e}")
                    return data, {"error": f"Key '{key}' not found in data dictionary"}
                except Exception as e:
                    logging.error(f"Error extracting scores from dictionary data: {e}")
                    return data, {"error": f"Error extracting scores: {str(e)}"}
            else:
                original_data = None
                try:
                    # Validate and clean raw data
                    cleaned_data = []
                    for value in data:
                        if value is None or (isinstance(value, float) and np.isnan(value)):
                            logging.warning(f"Skipping non-numeric value: {value}")
                            continue
                        if not isinstance(value, (int, float)):
                            logging.debug(f"Skipping non-numeric value of type {type(value)}: {value}")
                            continue
                        cleaned_data.append(float(value))

                    if not cleaned_data:
                        logging.error("No valid numeric values found in cleaned_data: %s", cleaned_data)
                        return data, {"error": "No valid numeric values found"}

                    scores = np.array(cleaned_data, dtype=np.float64)
                except Exception as e:
                    logging.error(f"Error creating array from data: {e}")
                    return data, {"error": f"Error creating array: {str(e)}"}

            # Verify we have a valid numeric array
            if not np.issubdtype(scores.dtype, np.number):
                logging.error("All values must be numeric. Found non-numeric values in data.")
                return data, {"error": "Non-numeric values found in data"}

            if len(scores) == 0:
                logging.error("No valid scores remain after filtering")
                return data, {"error": "No valid scores remain after filtering"}

            original_count = len(scores)

            # Calculate mean and standard deviation
            mean = np.mean(scores)
            std_dev = np.std(scores)

            # Calculate cutoff value (mean + std_threshold * std_dev)
            cutoff = mean + (std_threshold * std_dev)
            method_desc = f"standard deviation (keeping values < mean + {std_threshold} * std_dev)"

            # Gather statistics
            stats = {
                "original_count": original_count,
                "original_min": float(np.min(scores)),
                "original_max": float(np.max(scores)),
                "original_mean": float(mean),
                "original_std_dev": float(std_dev),
                "method": "std_dev",
                "threshold_value": std_threshold,
                "method_description": method_desc,
                "cutoff_value": float(cutoff),
            }

            # Filter based on the calculated cutoff
            if original_data is None:
                # Filter scores (keep values below the cutoff)
                filtered_scores = scores[scores <= cutoff]

                # Add filtered statistics
                stats.update(
                    {
                        "filtered_count": len(filtered_scores),
                        "removed_count": original_count - len(filtered_scores),
                        "filtered_min": (float(np.min(filtered_scores)) if len(filtered_scores) > 0 else None),
                        "filtered_max": (float(np.max(filtered_scores)) if len(filtered_scores) > 0 else None),
                        "filtered_mean": (float(np.mean(filtered_scores)) if len(filtered_scores) > 0 else None),
                    }
                )

                logging.info(f"Removed {stats['removed_count']} values above cutoff {cutoff:.2f}")
                return filtered_scores.tolist()[:5], stats

            # If we're working with dictionaries
            else:
                filtered_data = [item for item in original_data if item[key] <= cutoff]

                # Add filtered statistics
                if filtered_data:
                    filtered_scores = np.array([item[key] for item in filtered_data])
                    stats.update(
                        {
                            "filtered_count": len(filtered_data),
                            "removed_count": original_count - len(filtered_data),
                            "filtered_min": (float(np.min(filtered_scores)) if len(filtered_data) > 0 else None),
                            "filtered_max": (float(np.max(filtered_scores)) if len(filtered_data) > 0 else None),
                            "filtered_mean": (float(np.mean(filtered_scores)) if len(filtered_data) > 0 else None),
                        }
                    )
                else:
                    stats.update(
                        {
                            "filtered_count": 0,
                            "removed_count": original_count,
                            "filtered_min": None,
                            "filtered_max": None,
                            "filtered_mean": None,
                        }
                    )

                logging.info(f"Removed {stats['removed_count']} values above cutoff {cutoff:.2f}")
                return filtered_data, stats

        except Exception as e:
            logging.error(f"Error in remove_outliers: {str(e)}")
            return data, {"error": str(e)}

    def format_results(self, results: Optional[List[Dict[str, Any]]]) -> str:
        """Format search results for display"""
        if not results:
            return ""

        formatted_entries = []
        seen_texts = set()

        i = 1
        for result in results[0]:
            text = result["entity"]["text"]
            if text in seen_texts:
                continue

            seen_texts.add(text)
            entry = self._format_single_result(i, result["entity"])
            formatted_entries.append(entry)
            i += 1
        return "\n\n".join(formatted_entries)

    def _format_single_result(self, index: int, entity: Dict[str, Any]) -> str:
        """Format a single search result entry"""
        title = entity["title"].strip()
        text = entity["text"].replace(title, "").strip()
        source = entity["source"].strip()

        base_text = f"<small>{index}. [{title}]({source}), " f"_'{text}'_"

        return f"{base_text}</small>"
