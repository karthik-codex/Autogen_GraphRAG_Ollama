import tiktoken
import pandas as pd
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.query.input.loaders.dfs import store_entity_semantic_embeddings
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.global_search.community_context import GlobalCommunityContext
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.structured_search.local_search.search import LocalSearch

def initiate_graphRAG(commnuity, response_type, llm_config_graphRAG, token_encoder, text_embedder):
    INPUT_DIR = "input/artifacts"
    COMMUNITY_REPORT_TABLE = "create_final_community_reports"
    ENTITY_TABLE = "create_final_nodes"
    ENTITY_EMBEDDING_TABLE = "create_final_entities"
    RELATIONSHIP_TABLE = "create_final_relationships"
    TEXT_UNIT_TABLE = "create_final_text_units"
    LANCEDB_URI = f"{INPUT_DIR}/lancedb"
    COMMUNITY_LEVEL = commnuity
    RESPONSE_TYPE = response_type # Free form text e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report

    entity_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_TABLE}.parquet")
    entity_embedding_df = pd.read_parquet(f"{INPUT_DIR}/{ENTITY_EMBEDDING_TABLE}.parquet")
    entities = read_indexer_entities(entity_df, entity_embedding_df, COMMUNITY_LEVEL)

    description_embedding_store = LanceDBVectorStore(collection_name="entity_description_embeddings")
    description_embedding_store.connect(db_uri=LANCEDB_URI)
    store_entity_semantic_embeddings(entities=entities, vectorstore=description_embedding_store)

    relationship_df = pd.read_parquet(f"{INPUT_DIR}/{RELATIONSHIP_TABLE}.parquet")
    relationships = read_indexer_relationships(relationship_df)

    report_df = pd.read_parquet(f"{INPUT_DIR}/{COMMUNITY_REPORT_TABLE}.parquet")
    reports = read_indexer_reports(report_df, entity_df, COMMUNITY_LEVEL)

    text_unit_df = pd.read_parquet(f"{INPUT_DIR}/{TEXT_UNIT_TABLE}.parquet")
    text_units = read_indexer_text_units(text_unit_df)

    print(f"Report records: {len(report_df)}")
    report_df.head()

    context_builder = GlobalCommunityContext(
        community_reports=reports,
        entities=entities,
        token_encoder=token_encoder,
    )

    context_builder_local = LocalSearchMixedContext(
        community_reports=reports,
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        entity_text_embeddings=description_embedding_store,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=text_embedder,
        token_encoder=token_encoder,
    )

    ### 3. SETUP Context Builder for Global GraphRAG Search ###
    context_builder_params  = { "use_community_summary": False, "shuffle_data": True, "include_community_rank": True, 
                                "min_community_rank": 0, "community_rank_name": "rank", "include_community_weight": True, 
                                "community_weight_name": "occurrence weight", "normalize_community_weight": True, 
                                "max_tokens": 3_000, "context_name": "Reports", }
    map_llm_params          = { "max_tokens": 4000, "temperature": 0.0, "response_format": {"type": "json_object"}, }
    reduce_llm_params       = { "max_tokens": 2000, "temperature": 0.0, }

    search_engine = GlobalSearch(
        llm=llm_config_graphRAG,
        context_builder=context_builder,
        token_encoder=token_encoder,
        max_data_tokens=12_000,
        map_llm_params=map_llm_params,
        reduce_llm_params=reduce_llm_params,
        allow_general_knowledge=False,
        json_mode=True,
        context_builder_params=context_builder_params,
        concurrent_coroutines=10,
        response_type=RESPONSE_TYPE, # Free form text e.g. prioritized list, single paragraph, multiple paragraphs, multiple-page report
    )

    local_context_params = { "text_unit_prop": 0.5, "community_prop": 0.1, "conversation_history_max_turns": 5, 
                            "conversation_history_user_turns_only": True, "top_k_mapped_entities": 10, 
                            "top_k_relationships": 10, "include_entity_rank": True, "include_relationship_weight": True, 
                            "include_community_rank": False, "return_candidate_context": False, "max_tokens": 12_000, }
    llm_params          = { "max_tokens": 2_000, "temperature": 0.0, }

    search_engine_loc = LocalSearch(
        llm=llm_config_graphRAG,
        context_builder=context_builder_local,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
        response_type=RESPONSE_TYPE,
    )
    print(RESPONSE_TYPE)
    question_generator = LocalQuestionGen(
        llm=llm_config_graphRAG,
        context_builder=context_builder_local,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params,
    )

    return search_engine, search_engine_loc, question_generator