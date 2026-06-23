// Types mirroring the DocMind FastAPI response models.

export type DocumentStatus = "queued" | "processing" | "indexed" | "failed";

export interface DocumentRecord {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number;
  word_count: number;
  chunk_count: number;
  tags: string[];
  indexed_at: string;
  embedding_model: string;
  status: DocumentStatus;
  classification: string;
}

export interface SourceChunk {
  document_id: string;
  filename: string;
  page_number: number;
  chunk_index: number;
  char_offset: number;
  text: string;
  relevance_score: number;
  reranker_score?: number | null;
}

export type Confidence = "high" | "medium" | "low";

export interface SentenceSupport {
  text: string;
  supported: boolean;
  overlap: number;
}

export interface QueryResponse {
  query_id: string;
  session_id: string;
  answer: string;
  sources: SourceChunk[];
  confidence: Confidence;
  confidence_score: number;
  grounding_score?: number | null;
  sentence_support?: SentenceSupport[];
  retrieval_mode: string;
  llm_model: string;
  prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  latency_ms: number;
  cache_hit: boolean;
}

export interface AppSettings {
  llm_provider: string;
  llm_model: string;
  embedding_provider: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  chunk_strategy: string;
  retrieval_k: number;
  reranker_top_k: number;
  retrieval_mode: string;
  enable_query_expansion: boolean;
  enable_hyde: boolean;
  enable_reranking: boolean;
  enable_context_compression: boolean;
  enable_crag: boolean;
  enable_self_rag: boolean;
  cache_backend: string;
  cache_ttl_seconds: number;
  llm_temperature: number;
  llm_max_tokens: number;
  log_level: string;
}

export interface EvaluationResult {
  run_id: string;
  timestamp: string;
  engine: string;
  question_count: number;
  faithfulness: number;
  answer_relevancy: number;
  context_precision: number;
  context_recall: number;
  unsupported_sentence_ratio: number;
  recall_at_k: number;
  ndcg: number;
  mrr: number;
}

export interface DriftStatus {
  drift_detected: boolean;
  drift_score: number;
  threshold: number;
  message: string;
}
