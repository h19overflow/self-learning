# From Local to Global: A GraphRAG Approach to Query-Focused Summarization

Darren Edge1† Ha Trinh1† Newman Cheng2 Joshua Bradley2 Alex Chao3   
Apurva Mody3 Steven Truitt2 Dasha Metropolitansky1 Robert Osazuwa Ness1

# Jonathan Larson1

1Microsoft Research 2Microsoft Strategic Missions and Technologies 3Microsoft Office of the CTO

{daedge,trinhha,newmancheng,joshbradley,achao,moapurva, steventruitt,dasham,robertness,jolarso}@microsoft.com †These authors contributed equally to this work

# Abstract

The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs) to answer questions over private and/or previously unseen document collections. However, RAG fails on global questions directed at an entire text corpus, such as “What are the main themes in the dataset?”, since this is inherently a queryfocused summarization (QFS) task, rather than an explicit retrieval task. Prior QFS methods, meanwhile, do not scale to the quantities of text indexed by typical RAG systems. To combine the strengths of these contrasting methods, we propose GraphRAG, a graph-based approach to question answering over private text corpora that scales with both the generality of user questions and the quantity of source text. Our approach uses an LLM to build a graph index in two stages: first, to derive an entity knowledge graph from the source documents, then to pregenerate community summaries for all groups of closely related entities. Given a question, each community summary is used to generate a partial response, before all partial responses are again summarized in a final response to the user. For a class of global sensemaking questions over datasets in the 1 million token range, we show that GraphRAG leads to substantial improvements over a conventional RAG baseline for both the comprehensiveness and diversity of generated answers.

# 1 Introduction

Retrieval augmented generation (RAG) (Lewis et al., 2020) is an established approach to using LLMs to answer queries based on data that is too large to contain in a language model’s context window, meaning the maximum number of tokens (units of text) that can be processed by the LLM at once (Kuratov et al., 2024; Liu et al., 2023). In the canonical RAG setup, the system has access to a large external corpus of text records and retrieves a subset of records that are individually relevant to the query and collectively small enough to fit into the context window of the LLM. The LLM then generates a response based on both the query and the retrieved records (Baumel et al., 2018; Dang, 2006; Laskar et al., 2020; Yao et al., 2017). This conventional approach, which we collectively call vector RAG, works well for queries that can be answered with information localized within a small set of records. However, vector RAG approaches do not support sensemaking queries, meaning queries that require global understanding of the entire dataset, such as ”What are the key trends in how scientific discoveries are influenced by interdisciplinary research over the past decade?”

Sensemaking tasks require reasoning over “connections (which can be among people, places, and events) in order to anticipate their trajectories and act effectively” (Klein et al., 2006). LLMs such as GPT (Achiam et al., 2023; Brown et al., 2020), Llama (Touvron et al., 2023), and Gemini (Anil et al., 2023) excel at sensemaking in complex domains like scientific discovery (Microsoft, 2023) and intelligence analysis (Ranade and Joshi, 2023). Given a sensemaking query and a text with an implicit and interconnected set of concepts, an LLM can generate a summary that answers the query. The challenge, however, arises when the volume of data requires a RAG approach, since vector RAG approaches are unable to support sensemaking over an entire corpus.

In this paper, we present GraphRAG – a graph-based RAG approach that enables sensemaking over the entirety of a large text corpus. GraphRAG first uses an LLM to construct a knowledge graph, where nodes correspond to key entities in the corpus and edges represent relationships between those entities. Next, it partitions the graph into a hierarchy of communities of closely related entities, before using an LLM to generate community-level summaries. These summaries are generated in a bottom-up manner following the hierarchical structure of extracted communities, with summaries at higher levels of the hierarchy recursively incorporating lower-level summaries. Together, these community summaries provide global descriptions and insights over the corpus. Finally, GraphRAG answers queries through map-reduce processing of community summaries; in the map step, the summaries are used to provide partial answers to the query independently and in parallel, then in the reduce step, the partial answers are combined and used to generate a final global answer.

The GraphRAG method and its ability to perform global sensemaking over an entire corpus form the main contribution of this work. To demonstrate this ability, we developed a novel application of the LLM-as-a-judge technique (Zheng et al., 2024) suitable for questions targeting broad issues and themes where there is no ground-truth answer. This approach first uses one LLM to generate a diverse set of global sensemaking questions based on corpus-specific use cases, before using a second LLM to judge the answers of two different RAG systems using predefined criteria (defined in Section 3.3). We use this approach to compare GraphRAG to vector RAG on two representative real-world text datasets. Results show GraphRAG strongly outperforms vector RAG when using GPT-4 as the LLM.

GraphRAG is available as open-source software at https://github.com/microsoft/graphrag. In addition, versions of the GraphRAG approach are also available as extensions to multiple opensource libraries, including LangChain (LangChain, 2024), LlamaIndex (LlamaIndex, 2024), NebulaGraph (NebulaGraph, 2024), and Neo4J (Neo4J, 2024).

# 2 Background

# 2.1 RAG Approaches and Systems

RAG generally refers to any system where a user query is used to retrieve relevant information from external data sources, whereupon this information is incorporated into the generation of a response to the query by an LLM (or other generative AI model, such as a multi-media model). The query and retrieved records populate a prompt template, which is then passed to the LLM (Ram et al., 2023). RAG is ideal when the total number of records in a data source is too large to include in a single prompt to the LLM, i.e. the amount of text in the data source exceeds the LLM’s context window.

In canonical RAG approaches, the retrieval process returns a set number of records that are semantically similar to the query and the generated answer uses only the information in those retrieved records. A common approach to conventional RAG is to use text embeddings, retrieving records closest to the query in vector space where closeness corresponds to semantic similarity (Gao et al., 2023). While some RAG approaches may use alternative retrieval mechanisms, we collectively refer to the family of conventional approaches as vector RAG. GraphRAG contrasts with vector RAG in its ability to answer queries that require global sensemaking over the entire data corpus.

GraphRAG builds upon prior work on advanced RAG strategies. GraphRAG leverages summaries over large sections of the data source as a form of ”self-memory” (described in Cheng et al. 2024), which are later used to answer queries as in Mao et al. 2020). These summaries are generated in parallel and iteratively aggregated into global summaries, similar to prior techniques (Feng et al., 2023; Gao et al., 2023; Khattab et al., 2022; Shao et al., 2023; Su et al., 2020; Trivedi et al., 2022; Wang et al., 2024). In particular, GraphRAG is similar to other approaches that use hierarchical indexing to create summaries (similar to Kim et al. 2023; Sarthi et al. 2024). GraphRAG contrasts with these approaches by generating a graph index from the source data, then applying graph-based community detection to create a thematic partitioning of the data.

# 2.2 Using Knowledge Graphs with LLMs and RAG

Approaches to knowledge graph extraction from natural language text corpora include rulematching, statistical pattern recognition, clustering, and embeddings (Etzioni et al., 2004; Kim et al., 2016; Mooney and Bunescu, 2005; Yates et al., 2007). GraphRAG falls into a more recent body of research that use of LLMs for knowledge graph extraction (Ban et al., 2023; Melnyk et al., 2022; OpenAI, 2023; Tan et al., 2017; Trajanoska et al., 2023; Yao et al., 2023; Yates et al., 2007; Zhang et al., 2024a). It also adds to a growing body of RAG approaches that use a knowledge graph as an index (Gao et al., 2023). Some techniques use subgraphs, elements of the graph, or properties of the graph structure directly in the prompt (Baek et al., 2023; He et al., 2024; Zhang, 2023) or as factual grounding for generated outputs (Kang et al., 2023; Ranade and Joshi, 2023). Other techniques (Wang et al., 2023b) use the knowledge graph to enhance retrieval, where at query time an LLM-based agent dynamically traverses a graph with nodes representing document elements (e.g., passages, tables) and edges encoding lexical and semantical similarity or structural relationships. GraphRAG contrasts with these approaches by focusing on a previously unexplored quality of graphs in this context: their inherent modularity (Newman, 2006) and the ability to partition graphs into nested modular communities of closely related nodes (e.g., Louvain, Blondel et al. 2008; Leiden, Traag et al. 2019). Specifically, GraphRAG recursively creates increasingly global summaries by using the LLM to create summaries spanning this community hierarchy.

# 2.3 Adaptive benchmarking for RAG Evaluation

Many benchmark datasets for open-domain question answering exist, including HotPotQA (Yang et al., 2018), MultiHop-RAG (Tang and Yang, 2024), and MT-Bench (Zheng et al., 2024). However, these benchmarks are oriented towards vector RAG performance, i.e., they evaluate performance on explicit fact retrieval. In this work, we propose an approach for generating a set of questions for evaluating global sensemaking over the entirety of the corpus. Our approach is related to LLM methods that use a corpus to generate questions whose answers would be summaries of the corpus, such as in $\mathrm { X u }$ and Lapata (2021). However, in order to produce a fair evaluation, our method avoids generating the questions directly from the corpus itself (as an alternative implementation, one can use a subset of the corpus held out from subsequent graph extraction and answer evaluation steps).

Adaptive benchmarking refers to the process of dynamically generating evaluation benchmarks tailored to specific domains or use cases. Recent work has used LLMs for adaptive benchmarking to ensure relevance, diversity, and alignment with the target application or task (Yuan et al., 2024; Zhang et al., 2024b). In this work, we propose an adaptive benchmarking approach to generating global sensemaking queries for the LLM. Our approach builds on prior work in LLM-based persona generation, where the LLM is used to generate diverse and authentic sets of personas (Kosinski, 2024; Salminen et al., 2024; Shin et al., 2024). Our adaptive benchmarking procedure uses persona generation to create queries that are representative of real-world RAG system usage. Specifically, our approach uses the LLM to infer the potential users would use the RAG system and their use cases, which guide the generation of corpus-specific sensemaking queries.

# 2.4 RAG evaluation criteria

Our evaluation relies on the LLM to evaluate how well the RAG system answers the generated questions. Prior work has shown LLMs to be good evaluators of natural language generation, including work where LLMs evaluations were competitive with human evaluations (Wang et al., 2023a; Zheng et al., 2024). Some prior work proposes criteria for having LLMs quantify the quality of generated texts such as “fluency” (Wang et al., 2023a) Some of these criteria are generic to vector RAG systems and not relevant to global sensemaking, such as “context relevance”, “faithfulness”, and “answer relevance” (RAGAS, Es et al. 2023). Lacking a gold standard for evaluation, one can quantify relative performance for a given criterion by prompting the LLM to compare generations from two different competing models (LLM-as-a-judge, (Zheng et al., 2024)). In this work, we design criteria for evaluating RAG-generated answers to global sensemaking questions and evaluate our results using the comparative approach. We also validate results using statistics derived from LLM-extracted statements of verifiable facts, or “claims.”

![## Image Analysis: ef2268bf707b39dbec41e30deb0801c52ea63b336159bc302373d3668160d160.jpg

**Conceptual Understanding:**
This image conceptually represents the architecture and processing steps of a Graph Retrieval Augmented Generation (RAG) pipeline. Its main purpose is to illustrate how raw "Source Documents" are transformed into a structured "Knowledge Graph" during an "Indexing Time" phase, and subsequently, how this knowledge graph is utilized during a "Query Time" phase to generate a "Global Answer" to a given query by leveraging community detection and various summarization techniques. The key ideas communicated are the two distinct phases of graph RAG, the importance of domain-tailored and query-focused summarization, and the role of community detection in organizing the knowledge for efficient query answering.

**Content Interpretation:**
The image shows a multi-stage data processing pipeline.

**Processes being shown:**
*   **Data Preparation/Indexing:** "text extraction and chunking" transforms "Source Documents" into manageable "Text Chunks." This is the initial step for parsing raw data.
*   **Knowledge Representation:** "domain-tailored summarization" is applied twice. First, to extract "Entities & Relationships" from "Text Chunks," signifying the move from raw text to structured information. Second, to build a "Knowledge Graph" from "Entities & Relationships," indicating a further formalization and interlinking of information into a graph structure.
*   **Graph Partitioning:** "community detection" is a crucial process that takes the "Knowledge Graph" and breaks it down into "Graph Communities." This suggests an organization or clustering of related graph elements, likely for more efficient processing or localized summarization.
*   **Query Answering/Generation:**
    *   Another "domain-tailored summarization" step is applied to "Graph Communities" to create "Community Summaries," implying a contextual summary for each identified community.
    *   "query-focused summarization" is applied twice. First, to derive "Community Answers" from "Community Summaries," indicating summaries tailored to a specific query within the context of each community. Second, to synthesize these "Community Answers" into a final "Global Answer," suggesting an aggregation of relevant information across all communities to address the overall query.

**Significance of information presented:**
*   The distinction between "Indexing Time" and "Query Time" (explicitly labeled at the bottom) highlights the separation of concerns: one phase for building the knowledge base and another for utilizing it to answer queries. This is significant for understanding the operational flow and potential computational costs.
*   The repeated use of "domain-tailored summarization" emphasizes that the summarization processes are customized for the specific subject matter, implying a higher quality and relevance of extracted information and summaries.
*   "community detection" is a critical intermediate step that bridges the gap between the static "Knowledge Graph" and the dynamic "Query Time" phase. It signifies a strategy to manage complexity by breaking the graph into more manageable, semantically cohesive units, which is explicitly mentioned in the document context as enabling parallel summarization.
*   The progression from "Community Summaries" to "Community Answers" and finally to a "Global Answer" using "query-focused summarization" demonstrates a hierarchical and iterative approach to synthesizing information, ensuring that the final answer is comprehensive and directly addresses the user's query.

All extracted text elements directly support these interpretations by naming the inputs, outputs, and transformations at each stage, thus explicitly defining the processes and information types involved.

**Key Insights:**
**Main takeaways and insights:**
*   **Two-Phase Architecture:** The RAG pipeline is clearly divided into an "Indexing Time" phase for knowledge graph construction and a "Query Time" phase for answering queries, as evidenced by the explicit labels "Indexing Time" and "Query Time" and the distinct processing flows under each.
*   **Layered Summarization:** The process relies heavily on various forms of summarization: "text extraction and chunking" for initial processing, "domain-tailored summarization" for entity/relationship extraction and community summaries, and "query-focused summarization" for generating community and global answers. This indicates a multi-level approach to distil information.
*   **Graph-based Knowledge Representation:** The core of the indexed knowledge is a "Knowledge Graph," which implies that relationships and entities are explicitly represented, allowing for richer semantic understanding beyond simple text chunks. This is built from "Entities & Relationships" which are derived from "Text Chunks."
*   **Community-driven Query Resolution:** "community detection" is a pivotal step that organizes the "Knowledge Graph" into "Graph Communities." Query answering then proceeds by first summarizing these communities ("Community Summaries") and then generating "Community Answers" before synthesizing them into a "Global Answer" using "query-focused summarization." This suggests an efficient strategy for handling large knowledge graphs by processing relevant sub-sections (communities) in parallel, as mentioned in the accompanying text.
*   **Domain Specificity and Query Specificity:** The repeated use of "domain-tailored summarization" during indexing and for community summaries, coupled with "query-focused summarization" for deriving answers, highlights the pipeline's ability to provide contextually relevant and precise responses by adapting to both the domain of the source documents and the specifics of the user's query.

**Evidence from extracted text:**
*   "Indexing Time" and "Query Time" define the macro structure.
*   "Source Documents" to "Text Chunks" via "text extraction and chunking."
*   "Text Chunks" to "Entities & Relationships" to "Knowledge Graph" all use "domain-tailored summarization," demonstrating the graph construction.
*   "Knowledge Graph" to "Graph Communities" via "community detection" shows the partitioning.
*   "Graph Communities" to "Community Summaries" via "domain-tailored summarization."
*   "Community Summaries" to "Community Answers" to "Global Answer" all use "query-focused summarization," illustrating the answer generation process.

**Document Context:**
This image, titled "Figure 1: Graph RAG pipeline using an LLM-derived graph index of source document text" in the document context, perfectly illustrates the core mechanism being discussed in Section 2.4, "RAG evaluation criteria." It visually breaks down how a Retrieval Augmented Generation system, specifically one leveraging a graph-based knowledge representation, operates. It provides a detailed, step-by-step understanding of how an LLM can be used to process raw documents, build a structured graph index, and then query that index to generate comprehensive answers, directly supporting the technical discussion of RAG pipelines and their evaluation.

**Summary:**
The diagram outlines a "Graph RAG pipeline" which processes information in two main phases: "Indexing Time" and "Query Time," separated by a "Pipeline Stage" indicator.

During **Indexing Time**, the pipeline begins with "Source Documents." These documents are first subjected to "text extraction and chunking" to break them down into smaller, manageable "Text Chunks." Next, a "domain-tailored summarization" process extracts key "Entities & Relationships" from these text chunks. These entities and relationships are then further processed through another "domain-tailored summarization" step to construct a comprehensive "Knowledge Graph." This knowledge graph represents the structured understanding derived from the source documents.

The "Knowledge Graph" then undergoes "community detection," a process that identifies and groups related elements within the graph. This step effectively transitions the data from the indexing phase to the query phase by organizing the graph into "Graph Communities."

During **Query Time**, the pipeline utilizes these "Graph Communities." Each community is summarized using "domain-tailored summarization" to produce "Community Summaries." These summaries are then refined through "query-focused summarization" to generate "Community Answers," which are specific responses derived from individual communities in relation to a given query. Finally, all "Community Answers" are combined and synthesized using another round of "query-focused summarization" to produce the ultimate "Global Answer" to the user's query.

In essence, this pipeline efficiently transforms raw documents into a structured, queryable knowledge base and then systematically generates an answer by leveraging hierarchical summarization and community-based information retrieval. The consistent use of "domain-tailored" and "query-focused" summarization techniques highlights the system's ability to provide highly relevant and context-specific responses throughout the process.](images/ef2268bf707b39dbec41e30deb0801c52ea63b336159bc302373d3668160d160.jpg)
Figure 1: Graph RAG pipeline using an LLM-derived graph index of source document text. This graph index spans nodes (e.g., entities), edges (e.g., relationships), and covariates (e.g., claims) that have been detected, extracted, and summarized by LLM prompts tailored to the domain of the dataset. Community detection (e.g., Leiden, Traag et al., 2019) is used to partition the graph index into groups of elements (nodes, edges, covariates) that the LLM can summarize in parallel at both indexing time and query time. The “global answer” to a given query is produced using a final round of query-focused summarization over all community summaries reporting relevance to that query.

# 3 Methods

# 3.1 GraphRAG Workflow

Figure 1 illustrates the high-level data flow of the GraphRAG approach and pipeline. In this section, we describe the key design parameters, techniques, and implementation details for each step.

# 3.1.1 Source Documents Text Chunks

To start, the documents in the corpus are split into text chunks. The LLM extracts information from each chunk for downstream processing. Selecting the size of the chunk is a fundamental design decision; longer text chunks require fewer LLM calls for such extraction (which reduces cost) but suffer from degraded recall of information that appears early in the chunk (Kuratov et al., 2024; Liu et al., 2023). See Section A.1 for prompts and examples of the recall-precision trade-offs.

# 3.1.2 Text Chunks Entities & Relationships

In this step, the LLM is prompted to extract instances of important entities and the relationships between the entities from a given chunk. Additionally, the LLM generates short descriptions for the entities and relationships. To illustrate, suppose a chunk contained the following text:

NeoChip’s (NC) shares surged in their first week of trading on the NewTech Exchange. However, market analysts caution that the chipmaker’s public debut may not reflect trends for other technology IPOs. NeoChip, previously a private entity, was acquired by Quantum Systems in 2016. The innovative semiconductor firm specializes in low-power processors for wearables and IoT devices.

The LLM is prompted such that it extracts the following:

• The entity NeoChip, with description “NeoChip is a publicly traded company specializing in low-power processors for wearables and IoT devices.”   
• The entity Quantum Systems, with description “Quantum Systems is a firm that previously owned NeoChip.”   
• A relationship between NeoChip and Quantum Systems, with description “Quantum Systems owned NeoChip from 2016 until NeoChip became publicly traded.”

These prompts can be tailored to the domain of the document corpus by choosing domain appropriate few-shot exemplars for in-context learning (Brown et al., 2020). For example, while our default prompt extracts the broad class of “named entities” like people, places, and organizations and is generally applicable, domains with specialized knowledge (e.g., science, medicine, law) will benefit from few-shot exemplars specialized to those domains.

The LLM can also be prompted to extract claims about detected entities. Claims are important factual statements about entities, such as dates, events, and interactions with other entities. As with entities and relationships, in-context learning exemplars can provide domain-specific guidance. Claim descriptions extracted from the example tetx chunk are as follows:

• NeoChip’s shares surged during their first week of trading on the NewTech Exchange. • NeoChip debuted as a publicly listed company on the NewTech Exchange. • Quantum Systems acquired NeoChip in 2016 and held ownership until NeoChip went public.

See Appendix A for prompts and details on our implementation of entity and claim extraction.

# 3.1.3 Entities & Relationships Knowledge Graph

The use of an LLM to extract entities, relationships, and claims is a form of abstractive summarization – these are meaningful summaries of concepts that, in the case of relationships and claims, may not be explicitly stated in the text. The entity/relationship/claim extraction processes creates multiple instances of a single element because an element is typically detected and extracted multiple times across documents.

In the final step of the knowledge graph extraction process, these instances of entities and relationships become individual nodes and edges in the graph. Entity descriptions are aggregated and summarized for each node and edge. Relationships are aggregated into graph edges, where the number of duplicates for a given relationship becomes edge weights. Claims are aggregated similarly.

In this manuscript, our analysis uses exact string matching for entity matching – the task of reconciling different extracted names for the same entity (Barlaug and Gulla, 2021; Christen and Christen, 2012; Elmagarmid et al., 2006). However, softer matching approaches can be used with minor adjustments to prompts or code. Furthermore, GraphRAG is generally resilient to duplicate entities since duplicates are typically clustered together for summarization in subsequent steps.

# 3.1.4 Knowledge Graph Graph Communities

Given the graph index created in the previous step, a variety of community detection algorithms may be used to partition the graph into communities of strongly connected nodes (e.g., see the surveys by Fortunato (2010) and Jin et al. (2021)). In our pipeline, we use Leiden community detection (Traag et al., 2019) in a hierarchical manner, recursively detecting sub-communities within each detected community until reaching leaf communities that can no longer be partitioned.

Each level of this hierarchy provides a community partition that covers the nodes of the graph in a mutually exclusive, collectively exhaustive way, enabling divide-and-conquer global summarization. An illustration of such hierarchical partitioning on an example dataset can be found in Appendix B.

# 3.1.5 Graph Communities Community Summaries

The next step creates report-like summaries of each community in the community hierarchy, using a method designed to scale to very large datasets. These summaries are independently useful as a way to understand the global structure and semantics of the dataset, and may themselves be used to make sense of a corpus in the absence of a specific query. For example, a user may scan through community summaries at one level looking for general themes of interest, then read linked reports at a lower level that provide additional details for each subtopic. Here, however, we focus on their utility as part of a graph-based index used for answering global queries.

GraphRAG generates community summaries by adding various element summaries (for nodes, edges, and related claims) to a community summary template. Community summaries from lowerlevel communities are used to generate summaries for higher-level communities as follows:

• Leaf-level communities. The element summaries of a leaf-level community are prioritized and then iteratively added to the LLM context window until the token limit is reached. The prioritization is as follows: for each community edge in decreasing order of combined source and target node degree (i.e., overall prominence), add descriptions of the source node, target node, the edge itself, and related claims. Higher-level communities. If all element summaries fit within the token limit of the context window, proceed as for leaf-level communities and summarize all element summaries within the community. Otherwise, rank sub-communities in decreasing order of element summary tokens and iteratively substitute sub-community summaries (shorter) for their associated element summaries (longer) until they fit within the context window.

# 3.1.6 Community Summaries Community Answers Global Answer

Given a user query, the community summaries generated in the previous step can be used to generate a final answer in a multi-stage process. The hierarchical nature of the community structure also means that questions can be answered using the community summaries from different levels, raising the question of whether a particular level in the hierarchical community structure offers the best balance of summary detail and scope for general sensemaking questions (evaluated in section 4).

For a given community level, the global answer to any user query is generated as follows:

• Prepare community summaries. Community summaries are randomly shuffled and divided into chunks of pre-specified token size. This ensures relevant information is distributed across chunks, rather than concentrated (and potentially lost) in a single context window. • Map community answers. Intermediate answers are generated in parallel. The LLM is also asked to generate a score between 0-100 indicating how helpful the generated answer is in answering the target question. Answers with score 0 are filtered out. • Reduce to global answer. Intermediate community answers are sorted in descending order of helpfulness score and iteratively added into a new context window until the token limit is reached. This final context is used to generate the global answer returned to the user.

# 3.2 Global Sensemaking Question Generation

To evaluate the effectiveness of RAG systems for global sensemaking tasks, we use an LLM to generate a set of corpus-specific questions designed to asses high-level understanding of a given corpus, without requiring retrieval of specific low-level facts. Instead, given a high-level description of a corpus and its purposes, the LLM is prompted to generate personas of hypothetical users of the RAG system. For each hypothetical user, the LLM is then prompted to specify tasks that this user would use the RAG system to complete. Finally, for each combination of user and task, the LLM is prompted to generate questions that require understanding of the entire corpus. Algorithm 1 describes the approach.

# Algorithm 1: Prompting Procedure for Question Generation

1: Input: Description of a corpus, number of users $K$ , number of tasks per user $N$ , number of questions per (user, task) combination $M$ .   
2: Output: A set of $K * N * M$ high-level questions requiring global understanding of the corpus.   
3: procedure GENERATEQUESTIONS   
4: Based on the corpus description, prompt the LLM to: 1. Describe personas of $K$ potential users of the dataset. 2. For each user, identify $N$ tasks relevant to the user. 3. Specific to each user & task pair, generate $M$ high-level questions that: • Require understanding of the entire corpus. • Do not require retrieval of specific low-level facts.

Collect the generated questions to produce $K * N * M$ test questions for the dataset.

6: end procedure

For our evaluation, we set $K = M = N = 5$ for a total of 125 test questions per dataset. Table 1 shows example questions for each of the two evaluation datasets.

# 3.3 Criteria for Evaluating Global Sensemaking

Given the lack of gold standard answers to our activity-based sensemaking questions, we adopt the head-to-head comparison approach using an LLM evaluator that judges relative performance according to specific criteria. We designed three target criteria capturing qualities that are desirable for global sensemaking activities.

Appendix F shows the prompts for our head-to-head measures computed using an LLM evaluator, summarized as:

• Comprehensiveness. How much detail does the answer provide to cover all aspects and details of the question?   
• Diversity. How varied and rich is the answer in providing different perspectives and insights on the question?   
• Empowerment. How well does the answer help the reader understand and make informed   
judgments about the topic?

Table 1: Examples of potential users, tasks, and questions generated by the LLM based on short descriptions of the target datasets. Questions target global understanding rather than specific details.   

<table><tr><td>Dataset</td><td>Example activity framing and generation of global sensemaking questions</td></tr><tr><td>Podcast transcripts</td><td>User: A tech journalist looking for insights and trends in the tech industry Task: Understanding how tech leaders view the role of policy and regulation Questions: 1. Which episodes deal primarily with tech policy and government regulation? 2.How do guests perceive the impact of privacy laws on technology development?</td></tr><tr><td>News articles</td><td>4.What are the suggested changes to current policies mentioned by the guests? 5.Are collaborations between tech companies and governments discussed and how? User: Educator incorporating current affairs into curricula Task: Teaching about health and wellness Questions: 1. What current topics in health can be integrated into health education curricula? 2.How do news articles address the concepts of preventive medicine and wellness?</td></tr></table>

Furthermore, we use a “control criterion” called Directness that answers “How specifically and clearly does the answer address the question?”. In plain terms, directness evaluates the concision of an answer in a generic sense that applies to any generated LLM summarization. We include it to behave as a reference against which we can judge the soundness of results for the other criteria. Since directness is effectively in opposition to comprehensiveness and diversity, we would not expect any method to win across all four criteria.

In our evaluations, the LLM is provided with the question, the generated answers from two competing systems, and prompted to compare the two answers according to the criterion before giving a final judgment of which answer is preferred. The LLM either indicates a winner; or, it returns a tie if they are fundamentally similar. To account for the inherent stochasticity of LLM generation, we run each comparison with multiple replicates and average the results across replicates and questions. An illustration of LLM assessment for answers to a sample question can be found in Appendix D.

# 4 Analysis

# 4.1 Experiment 1

# 4.1.1 Datasets

We selected two datasets in the one million token range, each representative of corpora that users may encounter in their real-world activities:

Podcast transcripts. Public transcripts of Behind the Tech with Kevin Scott, a podcast featuring conversations between Microsoft CTO Kevin Scott and various thought leaders in science and technology (Scott, 2024). This corpus was divided into $1 6 6 9 \times 6 0 0$ -token text chunks, with 100-token overlaps between chunks ${ \sim } 1$ million tokens).

News articles. A benchmark dataset comprised of news articles published from September 2013 to December 2023 in a range of categories, including entertainment, business, sports, technology, health, and science (Tang and Yang, 2024). The corpus is divided into $3 1 9 7 \times 6 0 0$ -token text chunks, with 100-token overlaps between chunks ${ \sim } 1 . 7$ million tokens).

# 4.1.2 Conditions

We compared six conditions including GraphRAG at four different graph community levels (C0, C1, C2, C3), a text summarization method that applies our map-reduce approach directly to source texts (TS), and a vector RAG “semantic search” approach (SS):

• CO. Uses root-level community summaries (fewest in number) to answer user queries. • C1. Uses high-level community summaries to answer queries. These are sub-communities of C0, if present, otherwise C0 communities projected downwards. • C2. Uses intermediate-level community summaries to answer queries. These are subcommunities of C1, if present, otherwise C1 communities projected downwards. • C3. Uses low-level community summaries (greatest in number) to answer queries. These are sub-communities of C2, if present, otherwise C2 communities projected downwards. • TS. The same method as in Section 3.1.6, except source texts (rather than community summaries) are shuffled and chunked for the map-reduce summarization stages. • SS. An implementation of vector RAG in which text chunks are retrieved and added to the available context window until the specified token limit is reached.

The size of the context window and the prompts used for answer generation are the same across all six conditions (except for minor modifications to reference styles to match the types of context information used). Conditions only differ in how the contents of the context window are created.

The graph index supporting conditions C0-C3 was created using our generic prompts for entity and relationship extraction, with entity types and few-shot examples tailored to the domain of the data.

# 4.1.3 Configuration

We used a fixed context window size of 8k tokens for generating community summaries, community answers, and global answers (explained in Appendix C). Graph indexing with a 600 token window (explained in Section A.2) took 281 minutes for the Podcast dataset, running on a virtual machine (16GB RAM, Intel(R) Xeon(R) Platinum 8171M CPU $@$ 2.60GHz) and using a public OpenAI endpoint for gpt-4-turbo (2M TPM, 10k RPM).

We implemented Leiden community detection using the graspologic library (Chung et al., 2019). The prompts used to generate the graph index and global answers can be found in Appendix E, while the prompts used to evaluate LLM responses against our criteria can be found in Appendix F. A full statistical analysis of the results presented in the next section can be found in Appendix G.

# 4.2 Experiment 2

To validate the comprehensiveness and diversity results from Experiment 1, we implemented claimbased measures of these qualities. We use the definition of a factual claim from Ni et al. (2024), which is “a statement that explicitly presents some verifiable facts.” For example, the sentence “California and New York implemented incentives for renewable energy adoption, highlighting the broader importance of sustainability in policy decisions” contains two factual claims: (1) California implemented incentives for renewable energy adoption, and (2) New York implemented incentives for renewable energy adoption.

To extract factual claims, we used Claimify (Metropolitansky and Larson, 2025), an LLM-based method that identifies sentences in an answer containing at least one factual claim, then decomposes these sentences into simple, self-contained factual claims. We applied Claimify to the answers generated under the conditions from Experiment 1. After removing duplicate claims from each answer, we extracted 47,075 unique claims, with an average of 31 claims per answer.

We defined two metrics, with higher values indicating better performance:

1. Comprehensiveness: Measured as the average number of claims extracted from the answers generated under each condition.   
2. Diversity: Measured by clustering the claims for each answer and calculating the average number of clusters.

For clustering, we followed the approach described by Padmakumar and He (2024), which involved using Scikit-learn’s implementation of agglomerative clustering (Pedregosa et al., 2011). Clusters were merged through “complete” linkage, meaning they were combined only if the maximum distance between their farthest points was less than or equal to a predefined distance threshold. The distance metric used was $1 - \mathrm { R O U G E { - } L }$ . Since the distance threshold influences the number of clusters, we report results across a range of thresholds.

# 5 Results

# 5.1 Experiment 1

The indexing process resulted in a graph consisting of 8,564 nodes and 20,691 edges for the Podcast dataset, and a larger graph of 15,754 nodes and 19,520 edges for the News dataset. Table 2 shows the number of community summaries at different levels of each graph community hierarchy.

Global approaches vs. vector RAG. As shown in Figure 2 and Table 6, global approaches significantly outperformed conventional vector RAG (SS) in both comprehensiveness and diversity criteria across datasets. Specifically, global approaches achieved comprehensiveness win rates between 72- $83 \%$ $( \mathsf { p } { < } . 0 0 1 )$ for Podcast transcripts and $7 2 \mathrm { - } 8 0 \%$ $( \mathbf { p } { < } . 0 0 1 )$ ) for News articles, while diversity win rates ranged from $7 5 - 8 2 \%$ $( \mathsf { p } { < } . 0 0 1 )$ and $6 2 - 7 1 \%$ $\mathrm { ( p < . 0 1 ) }$ ) respectively. Our use of directness as a validity test confirmed that vector RAG produces the most direct responses across all comparisons.

Empowerment. Empowerment comparisons showed mixed results for both global approaches versus vector RAG (SS) and GraphRAG approaches versus source text summarization (TS). Using an LLM to analyze LLM reasoning for this measure indicated that the ability to provide specific exam

![## Image Analysis: 29e695197893607df2ac75e853c53c33926dd8dd8d2b47486a30282705a38b61.jpg

**Conceptual Understanding:**
This image conceptually represents a comprehensive performance comparison of different Retrieval-Augmented Generation (RAG) conditions. Its main purpose is to quantitatively demonstrate the relative strengths and weaknesses of various RAG configurations (labeled SS, TS, C0, C1, C2, C3) across different content types (Podcast transcripts, News articles) and desired output qualities (Comprehensiveness, Diversity, Empowerment, Directness). The core idea being communicated is the 'head-to-head win rate', where each value indicates how often one condition performs better than another in direct comparisons. This allows researchers to identify the most effective RAG strategies for specific evaluation criteria and data sources, highlighting superior performance through bolded percentages.

**Content Interpretation:**
The image systematically presents the performance evaluation of various Retrieval-Augmented Generation (RAG) conditions across two distinct datasets ('Podcast transcripts' and 'News articles') and four different evaluation metrics ('Comprehensiveness', 'Diversity', 'Empowerment', 'Directness'). Each 6x6 matrix details head-to-head win rates. The value in cell (R, C) represents the percentage of times condition R outperformed condition C. The conditions being compared are 'SS' (likely a baseline/self-comparison), 'TS' (global text summarization without a graph index), and 'C0', 'C1', 'C2', 'C3' (Graph RAG conditions). The diagonal elements (e.g., SS vs SS) are consistently 50%, representing a self-win rate, which serves as a neutral reference point. The bolded values within each matrix highlight the condition that achieved the highest win rate against all other conditions for that specific dataset and metric, indicating its superior performance. For instance, in 'Podcast transcripts' for 'Comprehensiveness', TS (83%), C0 (72%), C1 (75%), C2 (78%), and C3 (79%) all show high win rates against SS (17% to 28%), suggesting they are more 'comprehensive' than SS. The significance of the data lies in identifying which RAG conditions (C0-C3, TS) are more effective across different content types and performance dimensions.

**Key Insights:**
The main takeaways from this image are: 1. Graph RAG conditions (C0, C1, C2, C3) generally outperform the baseline/na¨ıve RAG (SS) across 'Comprehensiveness' and 'Diversity' for both 'Podcast transcripts' and 'News articles'. For example, in 'Podcast transcripts, Comprehensiveness', TS has an 83% win rate over SS, C0 has 72% over SS, C1 has 75% over SS, C2 has 78% over SS, and C3 has 79% over SS, all significantly above 50%. Similar trends are seen in 'Diversity' and 'News articles' datasets. 2. For 'Comprehensiveness' and 'Diversity', Graph RAG conditions C1, C2, C3, and TS often show strong performance, with TS being particularly strong against SS. 3. 'Empowerment' and 'Directness' metrics show more varied results, with 'SS' occasionally having high win rates against other conditions for these metrics, especially for 'Directness' against TS, C0, C1, C2, C3 in 'Podcast transcripts' (SS has bolded 56, 65, 60, 60, 60 against TS, C0, C1, C2, C3 respectively). This suggests that for 'Directness', 'SS' sometimes performs better or is perceived as more direct. 4. Conditions C1-C3 show improvements over TS in some scenarios, particularly in 'News articles, Comprehensiveness' where C1 (59% over TS), C2 (62% over TS), C3 (64% over TS) are notable, and in 'News articles, Diversity' where C2 (56% over TS) and C3 (60% over TS) stand out. These insights are directly supported by the bolded values and the percentages in the matrices, providing clear evidence for the relative strengths and weaknesses of each RAG condition under different contexts.

**Document Context:**
This image directly supports 'Section: 5.1 Experiment 1' by providing the quantitative results of an experiment designed to compare different RAG conditions. It illustrates the 'head-to-head win rate percentages' as described in the caption 'Figure 2'. The findings presented in these matrices, particularly the performance of Graph RAG conditions (C0-C3) compared to na¨ıve RAG (implied by SS's low win rates against C-conditions and TS) and global text summarization (TS), directly inform the discussion about which approaches are more effective for 'comprehensiveness and diversity' and 'answer comprehensiveness and diversity' as mentioned in the text after the image. The image provides the empirical evidence for the claims made, such as 'All Graph RAG conditions outperformed na¨ıve RAG on comprehensiveness and diversity' and 'Conditions C1-C3 also showed slight improvements in answer comprehensiveness and diversity over TS'.

**Summary:**
The image displays a series of matrices, organized into two main sections: 'Podcast transcripts' and 'News articles'. Each section contains four smaller matrices, each corresponding to a specific metric: 'Comprehensiveness', 'Diversity', 'Empowerment', and 'Directness'. These matrices present head-to-head win rate percentages, comparing different RAG conditions. The row headers and column headers for all matrices are 'SS', 'TS', 'C0', 'C1', 'C2', and 'C3'. Each cell [row condition, column condition] shows the win rate percentage of the row condition over the column condition. Self-win rates (diagonal elements) are consistently 50% as a reference. Bolded numbers within the matrices indicate the overall winner for that specific dataset and metric. The conditions 'C0', 'C1', 'C2', and 'C3' represent different Graph RAG conditions, while 'SS' likely stands for the 'Self-win' reference, and 'TS' represents 'global text summarization without a graph index' (as inferred from the document context for TS and naive RAG for SS).](images/29e695197893607df2ac75e853c53c33926dd8dd8d2b47486a30282705a38b61.jpg)
Figure 2: Head-to-head win rate percentages of (row condition) over (column condition) across two datasets, four metrics, and 125 questions per comparison (each repeated five times and averaged). The overall winner per dataset and metric is shown in bold. Self-win rates were not computed but are shown as the expected $50 \%$ for reference. All Graph RAG conditions outperformed na¨ıve RAG on comprehensiveness and diversity. Conditions C1-C3 also showed slight improvements in answer comprehensiveness and diversity over TS (global text summarization without a graph index).

Table 2: Number of context units (community summaries for C0-C3 and text chunks for TS), corresponding token counts, and percentage of the maximum token count. Map-reduce summarization of source texts is the most resource-intensive approach requiring the highest number of context tokens. Root-level community summaries (C0) require dramatically fewer tokens per query (9x-43x).   

<table><tr><td></td><td colspan="5">Podcast Transcripts</td><td colspan="5">News Articles</td></tr><tr><td></td><td>CO</td><td>C1</td><td>C2</td><td>C3</td><td>TS</td><td>C0</td><td>C1</td><td>C2</td><td>C3</td><td>TS</td></tr><tr><td>Units</td><td>34</td><td>367</td><td>969</td><td>1310</td><td>1669</td><td>55</td><td>555</td><td>1797</td><td>2142</td><td>3197</td></tr><tr><td>Tokens</td><td>26657</td><td>225756</td><td>565720</td><td>746100</td><td>1014611</td><td>39770</td><td>352641</td><td>980898</td><td></td><td>1140266 1707694</td></tr><tr><td>% Max</td><td>2.6</td><td>22.2</td><td>55.8</td><td>73.5</td><td>100</td><td>2.3</td><td>20.7</td><td>57.4</td><td>66.8</td><td>100</td></tr></table>

ples, quotes, and citations was judged to be key to helping users reach an informed understanding.   
Tuning element extraction prompts may help to retain more of these details in the GraphRAG index.

Community summaries vs. source texts. When comparing community summaries to source texts using GraphRAG, community summaries generally provided a small but consistent improvement in answer comprehensiveness and diversity, except for root-level summaries. Intermediate-level summaries in the Podcast dataset and low-level community summaries in the News dataset achieved comprehensiveness win rates of $57 \%$ $( \mathsf { p } { < } . 0 0 1 )$ and $64 \%$ $( \mathbf { p } { < } . 0 0 1 )$ , respectively. Diversity win rates were $57 \%$ $\left( \mathrm { p } { = } . 0 3 6 \right)$ for Podcast intermediate-level summaries and $60 \%$ $( \mathbf { p } { < } . 0 0 1 )$ for News low-level community summaries. Table 2 also illustrates the scalability advantages of GraphRAG compared to source text summarization: for low-level community summaries (C3), GraphRAG required 26- $33 \%$ fewer context tokens, while for root-level community summaries (C0), it required over $9 7 \%$ fewer tokens. For a modest drop in performance compared with other global methods, root-level GraphRAG offers a highly efficient method for the iterative question answering that characterizes sensemaking activity, while retaining advantages in comprehensiveness ( $72 \%$ win rate) and diversity ( $6 2 \%$ win rate) over vector RAG.

Table 3: Average number of extracted claims, reported by condition and dataset type. Bolded values represent the highest score in each column.   

<table><tr><td rowspan="2">Condition</td><td colspan="2">Average Number of Claims</td></tr><tr><td>News Articles</td><td>Podcast Transcripts</td></tr><tr><td>CO</td><td>34.18</td><td>32.21</td></tr><tr><td>C1</td><td>32.50</td><td>32.20</td></tr><tr><td>C2</td><td>31.62</td><td>32.46</td></tr><tr><td>C3</td><td>33.14</td><td>32.28</td></tr><tr><td>TS</td><td>32.89</td><td>31.39</td></tr><tr><td>SS</td><td>25.23</td><td>26.50</td></tr></table>

# 5.2 Experiment 2

Table 3 shows the results for the average number of extracted claims (i.e., the claim-based measure of comprehensiveness) per condition. For both the News and Podcast datasets, all global search conditions (C0-C3) and source text summarization (TS) had greater comprehensiveness than vector RAG (SS). The differences were statistically significant $( \mathsf { p } { < } . 0 5 )$ in all cases. These findings align with the LLM-based win rates from Experiment 1.

Table 4 contains the results for the average number of clusters, the claim-based measure of diversity. For the Podcast dataset, all global search conditions had significantly greater diversity than SS across all distance thresholds $( \mathsf { p } { < } . 0 5 )$ , consistent with the win rates observed in Experiment 1. For the News dataset, however, only C0 significantly outperformed SS across all distance thresholds $( \mathsf { p } { < } . 0 5 )$ . While C1-C3 also achieved higher average cluster counts than SS, the differences were statistically significant only at certain distance thresholds. In Experiment 1, all global search conditions significantly outperformed SS in the News dataset – not just C0. However, the differences in mean diversity scores between SS and the global search conditions were smaller for the News dataset than for the Podcast dataset, aligning directionally with the claim-based results.

For both comprehensiveness and diversity, across both datasets, there were no statistically significant differences observed among the global search conditions or between global search and TS.

Finally, for each pairwise comparison in Experiment 1, we tested whether the answer preferred by the LLM aligned with the winner based on the claim-based metrics. Since each pairwise comparison in Experiment 1 was performed five times, while the claim-based metrics provided only one outcome per comparison, we aggregated the Experiment 1 results into a single label using majority voting. For example, if C0 won over SS in three out of five judgments for comprehensiveness on a given question, C0 was labeled the winner and SS the loser. However, if C0 won twice, SS won once, and they tied twice, then there was no majority outcome, so the final label was a tie.

We found that exact ties were rare for the claim-based metrics. One possible solution is to define a tie based on a threshold (e.g., the absolute difference between the claim-based results for condition A and condition B must be less than or equal to $x$ ). However, we observed that the results were sensitive to the choice of threshold. As a result, we focused on cases where the aggregated LLM label was not a tie, representing $33 \%$ and $39 \%$ of pairwise comparisons for comprehensiveness and diversity, respectively. In these cases, the aggregated LLM label matched the claim-based label in $78 \%$ of pairwise comparisons for comprehensiveness and $6 9 - 7 0 \%$ for diversity (across all distance thresholds), indicating moderately strong alignment.

# 6 Discussion

# 6.1 Limitations of evaluation approach

Our evaluation to date has focused on sensemaking questions specific to two corpora each containing approximately 1 million tokens. More work is needed to understand how performance generalizes to datasets from various domains with different use cases. Comparison of fabrication rates, e.g., using approaches like SelfCheckGPT (Manakul et al., 2023), would also strengthen the current analysis.

Table 4: Average number of clusters across different distance thresholds, reported by condition and dataset type. Bolded values represent the highest score in each row.   

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Distance Threshold</td><td colspan="6">Average Number of Clusters</td></tr><tr><td>C0</td><td>C1</td><td>C2</td><td>C3</td><td>TS</td><td>SS</td></tr><tr><td rowspan="4">News Articles</td><td>0.5</td><td>23.42</td><td>21.85</td><td>21.90</td><td>22.13</td><td>21.80</td><td>17.92</td></tr><tr><td></td><td>21.65</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>06</td><td></td><td>20.38</td><td>20.30</td><td>20.5</td><td>20.13</td><td>16.78</td></tr><tr><td>0.8</td><td>18.86</td><td>17.78</td><td>17.82</td><td>17.79</td><td>17.30</td><td>14.80</td></tr><tr><td rowspan="4">Podcast Transcripts</td><td>0.5</td><td>23.16</td><td>22.62</td><td>22.52</td><td>21.93</td><td>21.14</td><td>18.55</td></tr><tr><td></td><td></td><td></td><td></td><td>20.62</td><td></td><td></td></tr><tr><td>06</td><td>21.45</td><td>21.3</td><td>21.21</td><td></td><td>19.70</td><td>17.39</td></tr><tr><td>0.8</td><td>19.26</td><td>18.77</td><td>18.46</td><td>17.89</td><td>16.66</td><td>15.07</td></tr></table>

# 6.2 Future work

The graph index, rich text annotations, and hierarchical community structure supporting the current GraphRAG approach offer many possibilities for refinement and adaptation. This includes RAG approaches that operate in a more local manner, via embedding-based matching of user queries and graph annotations. In particular, we see potential in hybrid RAG schemes that combine embeddingbased matching with just-in-time community report generation before employing our map-reduce summarization mechanisms. This “roll-up” approach could also be extended across multiple levels of the community hierarchy, as well as implemented as a more exploratory “drill down” mechanism that follows the information scent contained in higher-level community summaries.

Broader impacts. As a mechanism for question answering over large document collections, there are risks to downstream sensemaking and decision-making tasks if the generated answers do not accurately represent the source data. System use should be accompanied by clear disclosures of AI use and the potential for errors in outputs. Compared to vector RAG, however, GraphRAG shows promise as a way to mitigate these downstream risks for questions of a global nature, which might otherwise be answered by samples of retrieved facts falsely presented as global summaries.

# 7 Conclusion

We have presented GraphRAG, a RAG approach that combines knowledge graph generation and query-focused summarization (QFS) to support human sensemaking over entire text corpora. Initial evaluations show substantial improvements over a vector RAG baseline for both the comprehensiveness and diversity of answers, as well as favorable comparisons to a global but graph-free approach using map-reduce source text summarization. For situations requiring many global queries over the same dataset, summaries of root-level communities in the entity-based graph index provide a data index that is both superior to vector RAG and achieves competitive performance to other global methods at a fraction of the token cost.

# Acknowledgements

We would also like to thank the following people who contributed to the work: Alonso Guevara Fernandez, Amber Hoak, Andr ´ es Morales Esquivel, Ben Cutler, Billie Rinaldi, Chris Sanchez, ´ Chris Trevino, Christine Caggiano, David Tittsworth, Dayenne de Souza, Douglas Orbaker, Ed Clark, Gabriel Nieves-Ponce, Gaudy Blanco Meneses, Kate Lytvynets, Katy Smith, Monica Carva- ´ jal, Nathan Evans, Richard Ortega, Rodrigo Racanicci, Sarah Smith, and Shane Solomon.

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. (2023). Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.   
Baek, J., Aji, A. F., and Saffari, A. (2023). Knowledge-augmented language model prompting for zero-shot knowledge graph question answering. arXiv preprint arXiv:2306.04136.   
Ban, T., Chen, L., Wang, X., and Chen, H. (2023). From query tools to causal architects: Harnessing large language models for advanced causal discovery from data.   
Barlaug, N. and Gulla, J. A. (2021). Neural networks for entity matching: A survey. ACM Transactions on Knowledge Discovery from Data (TKDD), 15(3):1–37.   
Baumel, T., Eyal, M., and Elhadad, M. (2018). Query focused abstractive summarization: Incorporating query relevance, multi-document coverage, and summary length constraints into seq2seq models. arXiv preprint arXiv:1801.07704.   
Blondel, V. D., Guillaume, J.-L., Lambiotte, R., and Lefebvre, E. (2008). Fast unfolding of communities in large networks. Journal of statistical mechanics: theory and experiment, 2008(10):P10008.   
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. (2020). Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.   
Cheng, X., Luo, D., Chen, X., Liu, L., Zhao, D., and Yan, R. (2024). Lift yourself up: Retrievalaugmented text generation with self-memory. Advances in Neural Information Processing Systems, 36.   
Christen, P. and Christen, P. (2012). The data matching process. Springer.   
Chung, J., Pedigo, B. D., Bridgeford, E. W., Varjavand, B. K., Helm, H. S., and Vogelstein, J. T. (2019). Graspy: Graph statistics in python. Journal of Machine Learning Research, 20(158):1–7.   
Dang, H. T. (2006). Duc 2005: Evaluation of question-focused summarization systems. In Proceedings of the Workshop on Task-Focused Summarization and Question Answering, pages 48–55.   
Elmagarmid, A. K., Ipeirotis, P. G., and Verykios, V. S. (2006). Duplicate record detection: A survey. IEEE Transactions on knowledge and data engineering, 19(1):1–16.   
Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2023). Ragas: Automated evaluation of retrieval augmented generation. arXiv preprint arXiv:2309.15217.   
Etzioni, O., Cafarella, M., Downey, D., Kok, S., Popescu, A.-M., Shaked, T., Soderland, S., Weld, D. S., and Yates, A. (2004). Web-scale information extraction in knowitall: (preliminary results). In Proceedings of the 13th International Conference on World Wide Web, WWW $\cdot _ { 0 4 }$ , page 100–110, New York, NY, USA. Association for Computing Machinery.   
Feng, Z., Feng, X., Zhao, D., Yang, M., and Qin, B. (2023). Retrieval-generation synergy augmented large language models. arXiv preprint arXiv:2310.05149.   
Fortunato, S. (2010). Community detection in graphs. Physics reports, 486(3-5):75–174.   
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., and Wang, H. (2023). Retrievalaugmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.   
He, X., Tian, Y., Sun, Y., Chawla, N. V., Laurent, T., LeCun, Y., Bresson, X., and Hooi, B. (2024). G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv preprint arXiv:2402.07630.   
Huang, J., Chen, X., Mishra, S., Zheng, H. S., Yu, A. W., Song, X., and Zhou, D. (2023). Large language models cannot self-correct reasoning yet. arXiv preprint arXiv:2310.01798.   
Jacomy, M., Venturini, T., Heymann, S., and Bastian, M. (2014). Forceatlas2, a continuous graph layout algorithm for handy network visualization designed for the gephi software. PLoS ONE 9(6): e98679. https://doi.org/10.1371/journal.pone.0098679.   
Jin, D., Yu, Z., Jiao, P., Pan, S., He, D., Wu, J., Philip, S. Y., and Zhang, W. (2021). A survey of community detection approaches: From statistical modeling to deep learning. IEEE Transactions on Knowledge and Data Engineering, 35(2):1149–1170.   
Kang, M., Kwak, J. M., Baek, J., and Hwang, S. J. (2023). Knowledge graph-augmented language models for knowledge-grounded dialogue generation. arXiv preprint arXiv:2305.18846.   
Khattab, O., Santhanam, K., Li, X. L., Hall, D., Liang, P., Potts, C., and Zaharia, M. (2022). Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive nlp. arXiv preprint arXiv:2212.14024.   
Kim, D., Xie, L., and Ong, C. S. (2016). Probabilistic knowledge graph construction: Compositional and incremental approaches. In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, CIKM ’16, page 2257–2262, New York, NY, USA. Association for Computing Machinery.   
Kim, G., Kim, S., Jeon, B., Park, J., and Kang, J. (2023). Tree of clarifications: Answering ambiguous questions with retrieval-augmented large language models. arXiv preprint arXiv:2310.14696.   
Klein, G., Moon, B., and Hoffman, R. R. (2006). Making sense of sensemaking 1: Alternative perspectives. IEEE intelligent systems, 21(4):70–73.   
Kosinski, M. (2024). Evaluating large language models in theory of mind tasks. Proceedings of the National Academy of Sciences, 121(45):e2405460121.   
Kuratov, Y., Bulatov, A., Anokhin, P., Sorokin, D., Sorokin, A., and Burtsev, M. (2024). In search of needles in a 11m haystack: Recurrent memory finds what llms miss.   
LangChain (2024). Langchain graphs. https://langchain-graphrag.readthedocs.io/en/latest/.   
Laskar, M. T. R., Hoque, E., and Huang, J. (2020). Query focused abstractive summarization via incorporating query relevance and transfer learning with transformer models. In Advances in Artificial Intelligence: 33rd Canadian Conference on Artificial Intelligence, Canadian AI 2020, Ottawa, ON, Canada, May 13–15, 2020, Proceedings 33, pages 342–348. Springer.   
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, ¨ W.-t., Rocktaschel, T., et al. (2020). Retrieval-augmented generation for knowledge-intensive nlp ¨ tasks. Advances in Neural Information Processing Systems, 33:9459–9474.   
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. (2023). Lost in the middle: How language models use long contexts. arXiv:2307.03172.   
LlamaIndex (2024). GraphRAG Implementation with LlamaIndex - V2. https://github.com/runllama/llama index/blob/main/docs/docs/examples/cookbooks/GraphRAG v2.ipynb.   
Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. (2024). Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36.   
Manakul, P., Liusie, A., and Gales, M. J. (2023). Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.   
Mao, Y., He, P., Liu, X., Shen, Y., Gao, J., Han, J., and Chen, W. (2020). Generation-augmented retrieval for open-domain question answering. arXiv preprint arXiv:2009.08553.   
Martin, S., Brown, W. M., Klavans, R., and Boyack, K. (2011). Openord: An open-source toolbox for large graph layout. SPIE Conference on Visualization and Data Analysis (VDA).   
Melnyk, I., Dognin, P., and Das, P. (2022). Knowledge graph generation from text.

Metropolitansky, D. and Larson, J. (2025). Towards effective extraction and evaluation of factual claims.

Microsoft (2023). The impact of large language models on scientific discovery: a preliminary study using gpt-4.   
Mooney, R. J. and Bunescu, R. (2005). Mining knowledge from text using information extraction. SIGKDD Explor. Newsl., 7(1):3–10.   
NebulaGraph (2024). Nebulagraph launches industry-first graph rag: Retrieval-augmented generation with llm based on knowledge graphs. https://www.nebula-graph.io/posts/graph-RAG.   
Neo4J (2024). Get started with graphrag: Neo4j’s ecosystem tools. https://neo4j.com/developerblog/graphrag-ecosystem-tools/.   
Newman, M. E. (2006). Modularity and community structure in networks. Proceedings of the national academy of sciences, 103(23):8577–8582.   
Ni, J., Shi, M., Stammbach, D., Sachan, M., Ash, E., and Leippold, M. (2024). AFaCTA: Assisting the annotation of factual claim detection with reliable LLM annotators. In Ku, L.-W., Martins, A., and Srikumar, V., editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1890–1912, Bangkok, Thailand. Association for Computational Linguistics.   
OpenAI (2023). Chatgpt: Gpt-4 language model.   
Padmakumar, V. and He, H. (2024). Does writing with language models reduce content diversity? ICLR.   
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, E. (2011). Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12:2825–2830.   
Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., and Shoham, Y. (2023). In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331.   
Ranade, P. and Joshi, A. (2023). Fabula: Intelligence report generation using retrieval-augmented narrative construction. arXiv preprint arXiv:2310.13848.   
Salminen, J., Liu, C., Pian, W., Chi, J., Hayh ¨ anen, E., and Jansen, B. J. (2024). Deus ex machina ¨ and personas from large language models: Investigating the composition of ai-generated persona descriptions. In Proceedings of the CHI Conference on Human Factors in Computing Systems, pages 1–20.   
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., and Manning, C. D. (2024). Raptor: Recursive abstractive processing for tree-organized retrieval. arXiv preprint arXiv:2401.18059.   
Scott, K. (2024). Behind the Tech. https://www.microsoft.com/en-us/behind-the-tech.   
Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., and Chen, W. (2023). Enhancing retrievalaugmented large language models with iterative retrieval-generation synergy. arXiv preprint arXiv:2305.15294.   
Shin, J., Hedderich, M. A., Rey, B. J., Lucero, A., and Oulasvirta, A. (2024). Understanding humanai workflows for generating personas. In Proceedings of the 2024 ACM Designing Interactive Systems Conference, pages 757–781.   
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. (2024). Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36.   
Su, D., Xu, Y., Yu, T., Siddique, F. B., Barezi, E. J., and Fung, P. (2020). Caire-covid: A question answering and query-focused multi-document summarization system for covid-19 scholarly information management. arXiv preprint arXiv:2005.03975.   
Tan, Z., Zhao, X., and Wang, W. (2017). Representation learning of large-scale knowledge graphs via entity feature combinations. In Proceedings of the 2017 ACM on Conference on Information and Knowledge Management, CIKM ’17, page 1777–1786, New York, NY, USA. Association for Computing Machinery.   
Tang, Y. and Yang, Y. (2024). MultiHop-RAG: Benchmarking retrieval-augmented generation for multi-hop queries. arXiv preprint arXiv:2401.15391.   
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al. (2023). Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.   
Traag, V. A., Waltman, L., and Van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. Scientific Reports, 9(1).   
Trajanoska, M., Stojanov, R., and Trajanov, D. (2023). Enhancing knowledge graph construction using large language models. ArXiv, abs/2305.04676.   
Trivedi, H., Balasubramanian, N., Khot, T., and Sabharwal, A. (2022). Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. arXiv preprint arXiv:2212.10509.   
Wang, J., Liang, Y., Meng, F., Sun, Z., Shi, H., Li, Z., Xu, J., Qu, J., and Zhou, J. (2023a). Is chatgpt a good nlg evaluator? a preliminary study. arXiv preprint arXiv:2303.04048.   
Wang, S., Khramtsova, E., Zhuang, S., and Zuccon, G. (2024). Feb4rag: Evaluating federated search in the context of retrieval augmented generation. arXiv preprint arXiv:2402.11891.   
Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., and Zhou, D. (2022). Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.   
Wang, Y., Lipka, N., Rossi, R. A., Siu, A., Zhang, R., and Derr, T. (2023b). Knowledge graph prompting for multi-document question answering.   
Xu, Y. and Lapata, M. (2021). Text summarization with latent queries. arXiv preprint arXiv:2106.00104.   
Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., and Manning, C. D. (2018). HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP).   
Yao, J.-g., Wan, X., and Xiao, J. (2017). Recent advances in document summarization. Knowledge and Information Systems, 53:297–336.   
Yao, L., Peng, J., Mao, C., and Luo, Y. (2023). Exploring large language models for knowledge graph completion.   
Yates, A., Banko, M., Broadhead, M., Cafarella, M., Etzioni, O., and Soderland, S. (2007). TextRunner: Open information extraction on the web. In Carpenter, B., Stent, A., and Williams, J. D., editors, Proceedings of Human Language Technologies: The Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-HLT), pages 25–26, Rochester, New York, USA. Association for Computational Linguistics.   
Yuan, X., Li, J., Wang, D., Chen, Y., Mao, X., Huang, L., Xue, H., Wang, W., Ren, K., and Wang, J. (2024). S-eval: Automatic and adaptive test generation for benchmarking safety evaluation of large language models. arXiv preprint arXiv:2405.14191.   
Zhang, J. (2023). Graph-toolformer: To empower llms with graph reasoning ability via prompt augmented by chatgpt. arXiv preprint arXiv:2304.11116.   
Zhang, Y., Zhang, Y., Gan, Y., Yao, L., and Wang, C. (2024a). Causal graph discovery with retrievalaugmented generation based large language models. arXiv preprint arXiv:2402.15301.   
Zhang, Z., Chen, J., and Yang, D. (2024b). Darg: Dynamic evaluation of large language models via adaptive reasoning graph. arXiv preprint arXiv:2406.17271.   
Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. (2024). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36.   
Zhu, Y., Wang, X., Chen, J., Qiao, S., Ou, Y., Yao, Y., Deng, S., Chen, H., and Zhang, N. (2024). Llms for knowledge graph construction and reasoning: Recent capabilities and future opportunities.

# A Entity and Relationship Extraction Approach

The following prompts, designed for GPT-4, are used in the default GraphRAG initialization pipeline:

• Default Graph Extraction Prompt • Claim Extraction Prompt

# A.1 Entity Extraction

We do this using a multipart LLM prompt that first identifies all entities in the text, including their name, type, and description, before identifying all relationships between clearly related entities, including the source and target entities and a description of their relationship. Both kinds of element instance are output in a single list of delimited tuples.

# A.2 Self-Reflection

The choice of prompt engineering techniques has a strong impact on the quality of knowledge graph extraction (Zhu et al., 2024), and different techniques have different costs in terms of tokens consumed and generated by the model. Self-reflection is a prompt engineering technique where the LLM generates an answer, and is then prompted to evaluate its output for correctness, clarity, or completeness, then finally generate an improved response based on that evaluation (Huang et al., 2023; Madaan et al., 2024; Shinn et al., 2024; Wang et al., 2022). We leverage self-reflection in knowledge graph extraction, and explore ways how removing self-reflection affects performance and cost.

Using larger chunk size is less costly in terms of calls to the LLM. However, the LLM tends to extract few entities from chunks of larger size. For example, in a sample dataset (HotPotQA, Yang et al., 2018), GPT-4 extracted almost twice as many entity references when the chunk size was 600 tokens than when it was 2400. To address this issue, we deploy a self-reflection prompt engineering approach. After entities are extracted from a chunk, we provide the extracted entities back to the LLM, prompting it to “glean” any entities that it may have missed. This is a multi-stage process in which we first ask the LLM to assess whether all entities were extracted, using a logit bias of 100 to force a yes/no decision. If the LLM responds that entities were missed, then a continuation indicating that “MANY entities were missed in the last extraction” encourages the LLM to detect these missing entities. This approach allows us to use larger chunk sizes without a drop in quality (Figure 3) or the forced introduction of noise. We interate self-reflection steps up to a specified maximum number of times.

![## Image Analysis: 0d49c399116f4d55fe978dd19b43668e356117fbb7fc9bd2a03e14f0a7dc2b77.jpg

**Conceptual Understanding:**
This image conceptually represents an experimental evaluation of a machine learning or natural language processing model's performance in 'Entity references detected'. Its main purpose is to demonstrate how two key parameters—'Number of self-reflection iterations performed' and 'chunk size'—impact the model's ability to identify entities. The graph illustrates the positive correlation between self-reflection iterations and entity detection, and highlights the effect of chunk size on this performance, suggesting an optimal range or trade-off for these parameters in entity extraction tasks.

**Content Interpretation:**
The graph displays the performance of an entity extraction prompt (likely using gpt-4-turbo, as per the document context) in detecting entity references as a function of 'Number of self-reflection iterations performed' and 'chunk size'. It shows that increasing self-reflection iterations generally leads to a higher number of 'Entity references detected'. It also reveals that a smaller 'chunk size' (e.g., '600 chunk size') consistently results in more 'Entity references detected' compared to larger chunk sizes ('1200 chunk size' and '2400 chunk size') across all iteration counts. The lines show a monotonic increase, indicating that self-reflection is beneficial for entity detection. The different slopes suggest that the rate of improvement with iterations might vary slightly with chunk size, though all show a positive trend.

**Key Insights:**
The main takeaways are: 1. Increasing the 'Number of self-reflection iterations performed' generally leads to a higher count of 'Entity references detected'. 2. Smaller 'chunk size' settings (e.g., '600 chunk size') are more effective in detecting a greater number of entity references compared to larger chunk sizes ('1200 chunk size', '2400 chunk size') under the tested conditions. 3. The benefit of self-reflection iterations on entity detection is observed across all tested chunk sizes, as evidenced by the upward slope of all three lines. 4. The '600 chunk size' consistently outperforms the other two chunk sizes in terms of raw number of entity references detected at every iteration count.

**Document Context:**
The image, titled 'Figure 3: How the entity references detected in the HotPotQA dataset (Yang et al., 2018) varies with chunk size and self-reflection iterations for our generic entity extraction prompt with gpt-4-turbo' from Section A.2 Self-Reflection, directly supports the document's discussion on the efficacy of self-reflection and chunk sizing in natural language processing tasks, specifically entity extraction. It provides empirical evidence to evaluate how these parameters influence the performance of a generic entity extraction prompt, likely in the context of the HotPotQA dataset. This figure is crucial for understanding the practical implications and optimization strategies for entity detection using advanced language models like gpt-4-turbo.

**Summary:**
This image is a line graph illustrating the relationship between the number of self-reflection iterations performed and the quantity of entity references detected, for different chunk sizes. The x-axis, labeled 'Number of self-reflection iterations performed', ranges from 0 to 3 in integer steps. The y-axis, labeled 'Entity references detected', ranges from 0 to 30000, with major tick marks at 10000, 20000, and 30000. Three distinct lines represent different 'chunk size' values, identified by a legend in the top-left corner: '600 chunk size' (blue line with circular markers), '1200 chunk size' (green line with circular markers), and '2400 chunk size' (orange line with circular markers). Each line shows an upward trend, indicating that as the number of self-reflection iterations increases, the number of entity references detected also generally increases across all chunk sizes. The '600 chunk size' line consistently shows the highest number of detected entity references at each iteration count, followed by '1200 chunk size', and then '2400 chunk size'. Specifically, at 0 iterations, the detected entity references are approximately 9,500 for '600 chunk size', 7,000 for '1200 chunk size', and 6,000 for '2400 chunk size'. At 1 iteration, these values rise to about 16,000, 12,000, and 10,500, respectively. By 2 iterations, the values are around 19,500, 17,500, and 14,500. Finally, at 3 iterations, the '600 chunk size' reaches approximately 26,500 detected entity references, '1200 chunk size' reaches about 22,000, and '2400 chunk size' reaches approximately 19,500. This graph demonstrates a positive correlation between self-reflection iterations and detected entities, with smaller chunk sizes generally yielding more entity detections.](images/0d49c399116f4d55fe978dd19b43668e356117fbb7fc9bd2a03e14f0a7dc2b77.jpg)
Figure 3: How the entity references detected in the HotPotQA dataset (Yang et al., 2018) varies with chunk size and self-reflection iterations for our generic entity extraction prompt with gpt-4-turbo.

# B Example Community Detection

![## Image Analysis: 871844aa36a5e777d382c06fbacf6ddeb16a24089dde8ad2898b5993794c2fd1.jpg

**Conceptual Understanding:**
This image represents a visual output of a community detection algorithm applied to a network dataset. Conceptually, it illustrates how a large network of entities can be partitioned into meaningful subgroups or 'communities' based on their interconnections. The main purpose is to demonstrate the 'Root communities at level 0' – the primary, high-level clusters found within the MultiHop-RAG dataset using the Leiden algorithm. It visually conveys the inherent structure and organization of the entities within this dataset, highlighting clusters of highly connected components.

**Content Interpretation:**
The image depicts a network graph illustrating community detection results. It shows entities as nodes and their relationships as edges. The 'communities' are groups of closely related entities, visualized by distinct colors. The varying sizes of the nodes highlight their importance or centrality within the network (degree). The interpretation specifically focuses on 'Root communities at level 0', implying an initial, broad level of community organization detected hierarchically. The underlying algorithms (Leiden, OpenORD, Force Atlas 2) and dataset (MultiHop-RAG) are specified, providing a technical context for the visualization.

**Key Insights:**
1. The image visualizes entity communities: The different colors of the nodes explicitly show distinct communities among entities. 
2. Node size indicates degree: The size of each circle is directly proportional to its degree, meaning larger nodes are more connected. 
3. Hierarchical community structure: The caption 'Root communities at level 0' indicates that community detection is presented hierarchically, with this image showing the highest level of aggregation. 
4. Specific algorithms and dataset used: The image context (from the surrounding document) specifies the Leiden algorithm for detection, and OpenORD/Force Atlas 2 for layout, applied to the MultiHop-RAG dataset. These details provide the methodological basis for the visualization.

**Document Context:**
This image serves as a visual example for the 'B Example Community Detection' section of the document. It directly demonstrates the output of applying the Leiden algorithm for community detection to the MultiHop-RAG dataset. The subsequent text in the document explains that this particular image (a) shows Level 0 communities, establishing the foundational clustering before detailing further hierarchical levels, such as Level 1 (not shown in this specific image). It provides concrete evidence and visualization for the theoretical or methodological discussion of community detection in the document.

**Summary:**
This image displays a complex network graph, representing 'Root communities at level 0'. It illustrates the results of community detection using the Leiden algorithm on the MultiHop-RAG dataset. The graph is composed of numerous interconnected nodes, represented by circles of varying sizes, and lines representing connections between them. The nodes are colored differently to indicate distinct entity communities. The size of each circle is proportional to its degree, meaning larger circles represent nodes with more connections. The overall structure shows several densely connected clusters of similarly colored nodes, signifying the detected communities at the root level (Level 0) of hierarchical clustering, which corresponds to the partition with maximum modularity. The layout of the nodes has been performed using OpenORD and Force Atlas 2.](images/871844aa36a5e777d382c06fbacf6ddeb16a24089dde8ad2898b5993794c2fd1.jpg)
Figure 4: Graph communities detected using the Leiden algorithm (Traag et al., 2019) over the MultiHop-RAG (Tang and Yang, 2024) dataset as indexed. Circles represent entity nodes with size proportional to their degree. Node layout was performed via OpenORD (Martin et al., 2011) and Force Atlas 2 (Jacomy et al., 2014). Node colors represent entity communities, shown at two levels of hierarchical clustering: (a) Level 0, corresponding to the hierarchical partition with maximum modularity, and (b) Level 1, which reveals internal structure within these root-level communities.

![## Image Analysis: 2d8e900f9fbd70f796088b06383035cae1f65ad3eca364fb96e3889bdc429a14.jpg

**Conceptual Understanding:**
The image conceptually represents a network graph where nodes are individuals or entities and edges are connections or relationships between them. The main purpose is to illustrate the result of a 'community detection' algorithm, which groups highly interconnected nodes into distinct 'sub-communities.' The key idea communicated is that complex networks are not monolithic but often comprise smaller, cohesive groups that share common characteristics or interactions. The text '(b) Sub-communities at level 1' specifically labels these identified groups as the 'level 1' communities, indicating a primary or initial level of hierarchical decomposition in the network structure.

**Content Interpretation:**
The image represents a complex network, likely a social network, information network, or another form of interconnected system, where individual entities (nodes) are linked by relationships (edges). The primary concept being shown is 'community detection' at a 'level 1' granularity. The different colors assigned to various clusters of nodes and edges indicate distinct 'sub-communities.' These sub-communities are groups of nodes that are more densely connected to each other than to nodes outside their group. The varied colors, such as light green, blue, pink, purple, red, yellow, and black, visually segment the network into these identified communities. For instance, a prominent light green cluster is visible on the left, a blue cluster towards the top-right, and other colored clusters distributed throughout the graph. The presence of larger nodes within some clusters suggests potential 'hub' nodes or more central members within those specific communities. The text '(b) Sub-communities at level 1' explicitly states that this visualization shows the outcome of an algorithm or analysis that has identified these communities, specifically at a foundational or first-level decomposition.

**Key Insights:**
The main takeaway from this image is the successful visualization of distinct sub-communities within a larger, complex network. The visual evidence of differently colored clusters, supported by the textual label '(b) Sub-communities at level 1,' demonstrates that network analysis techniques can effectively identify hidden structures and groupings. This implies that even in a seemingly chaotic network, there are often inherent patterns and cohesive subgroups. The 'level 1' designation suggests a hierarchical approach to community detection, where different levels of granularity can be explored to understand network organization. This image provides insight into the practical outcome of community detection, showing how it segments a network into more manageable and interpretable components, each representing a tightly knit group.

**Document Context:**
This image is presented within a document section titled 'B Example Community Detection.' Therefore, the image serves as a direct visual example of the output or result of a community detection process applied to a network. It concretely illustrates how a complex, undifferentiated network can be analyzed to reveal underlying structural groups or 'sub-communities.' The caption '(b) Sub-communities at level 1' further contextualizes this by indicating that this is a specific hierarchical level of community detection, implying that further levels (e.g., level 2, level 3) might exist to reveal finer or broader groupings. This visualization helps readers understand the practical application and visual representation of community detection algorithms in real-world or theoretical networks.

**Summary:**
The image displays a highly complex network graph consisting of numerous interconnected nodes and edges, visualized in a circular, almost spherical, arrangement. The nodes are small dots, and the edges are fine lines connecting them, rendered in various colors such as light green, blue, pink, purple, red, yellow, and black. These colors are not uniform across the entire graph but rather appear to form distinct clusters or groups, indicating different communities within the network. Some nodes are slightly larger and more prominent, often appearing at the centers or key connection points within these colored clusters, suggesting higher importance or centrality. The overall density of connections is very high towards the center of the graph, gradually becoming sparser towards the periphery, where smaller, more isolated groups of nodes and edges are visible. The entire network is set against a plain white background, and the only textual element present is a caption centered below the graph. This image illustrates the concept of identifying and visualizing sub-communities within a larger network at a specific hierarchical level, demonstrating how a complex system can be decomposed into more manageable, related groups.](images/2d8e900f9fbd70f796088b06383035cae1f65ad3eca364fb96e3889bdc429a14.jpg)

# C Context Window Selection

The effect of context window size on any particular task is unclear, especially for models like gpt-4-turbo with a large context size of $1 2 8 \mathrm { k }$ tokens. Given the potential for information to be “lost in the middle” of longer contexts (Kuratov et al., 2024; Liu et al., 2023), we wanted to explore the effects of varying the context window size for our combinations of datasets, questions, and metrics. In particular, our goal was to determine the optimum context size for our baseline condition (SS) and then use this uniformly for all query-time LLM use. To that end, we tested four context window sizes: $^ \mathrm { 8 k }$ , 16k, 32k and $6 4 \mathrm { k }$ . Surprisingly, the smallest context window size tested $( 8 \mathbf { k } )$ was universally better for all comparisons on comprehensiveness (average win rate of $5 8 . 1 \%$ ), while performing comparably with larger context sizes on diversity (average win rate $= 5 2 . 4 \%$ ), and empowerment (average win rate $= 5 1 . 3 \%$ ). Given our preference for more comprehensive and diverse answers, we therefore used a fixed context window size of 8k tokens for the final evaluation.

# D Example Answer Comparison

Table 5: Example question, answers, and LLM-generated assessments for the News article dataset.   

<table><tr><td>Question</td><td>Which public figures are repeatedly mentioned across various entertainment articles?</td></tr><tr><td>Graph RAG</td><td>The entertainment industry is vast and diverse,encompassing film,television,music,sports,and digital media. Certain public figures stand out due to their significant contributions and influence across these sectors. The following summary highlights key individuals who are repeatedly mentioned in various entertainment articles, reflecting their impact and presence within the industry. Actors and Directors [..] Public Figures in Controversy [.] Musicians and Executives[.] Athletes and Coaches [..] Influencers and Entrepreneurs [..] The repeated mention of these figures in entertainment articles signifies their ongoing relevance and the public&#x27;s interest in their work.Their influence spans across various aspects of entertainment, from shaping cultural narratives in film and television to driving trends in music and digital media. These individuals not only contribute to their respective fields but also influence the broader cultural</td></tr><tr><td>Naive RAG</td><td>landscape,often becoming central figures in social discussions and public discourse. Public figures who are repeatedly mentioned across various entertainment articles include Taylor Swift,Travis Kelce, Britney Spears,and Justin Timberlake. These individuals have been highlighted for various reasons,ranging from their professional achievements to their personal lives. Taylor Swift [..] Travis Kelce[.] Britney Spears[.] Justin Timberlake [..] These figures are frequently covered due to their high-profile status and the public&#x27;s interest in their</td></tr><tr><td rowspan="4">LLM Decision</td><td>carers and personal lives. Their activities, whether in music,sports, or personal relationships, have significant cultural and economic impacts,as evidenced by the media coverage and public reactions. Comprehensiveness: Winner=1 (Graph RAG). Answer 1 is beter because it provides a more comprehensive and detailed list of public figures from a wider range of entertainment sectors, including film, television, music,sports,gaming,and digital media. It also includes examples of their contributions and the impact they have on their respective fields,as wel as mentions of controversies and their implications. Answer 2, while detailed in its coverage of a few individuals, is limited to a smaller number of public figures and focuses primarily on their personal lives and relationships rather</td></tr><tr><td>than a broad spectrum of their professional influence across the entertainment industry. Diversity: Winner=1(Graph RAG). Answer 1 is better because it provides a more varied and rich response by covering a wide range of public figures from different sectors of the entertainment industry,including film, television, music, sports,gaming,and digital media. It oers insights into the contributions and influence of these figures,as well as controversies and their impact on public discourse. The answer also cites specific data sources for each mentioned figure,indicating a diverse range of evidence to support the claims. In contrast, Answer 2 focuses on a smaller group of public figures, primarily from the music industry and sports,and relies heavily on a single source for data, which makes it less diverse in perspectives and insights.</td></tr><tr><td>Empowerment: Winner=1 (Graph RAG). Answer 1 is beter because it provides a comprehensive and structured overview of public figures across various sectors of the entertainment industry, including film, television, music,sports,and digital media.It lists multiple individuals, providing specific examples of their contributions and the context in which they are mentioned in entertainment articles,along with references to data reports for each claim.This approach helps the reader understand the breadth of the topic and make informed judgments without being misled.In contrast,</td></tr><tr><td>Answer 2 focuses on a smaller group of public figures and primarily discusses their personal lives and relationships,which may not provide as broad an understanding of the topic.While Answer 2 also cites sources,it does not match the depth and variety of Answer 1. Directness: Winner=2 (Naive RAG). Answer 2 is better because it directly lists specific public figures who are repeatedly mentioned across various entertainment articles,such as Taylor Swift, Travis Kelce, Britney Spears,and Justin Timberlake,and provides concise explanations for their frequent mentions.Answer 1,while comprehensive,includes a lot of detailed information about</td></tr></table>

# E System Prompts

# E.1 Element Instance Generation

---Goal--- Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

---Steps---

1. Identify all entities. For each identified entity, extract the following information: - entity name: Name of the entity, capitalized   
- entity type: One of the following types: [{entity types}]   
- entity description: Comprehensive description of the entity’s attributes and activities

Format each entity as ("entity"{tuple delimiter}<entity name>{tuple delimiter}<entity type>{tuple delimiter}<entity description>

2. From the entities identified in step 1, identify all pairs of (source entity, target entity) that   
are \*clearly related\* to each other   
For each pair of related entities, extract the following information:   
- source entity: name of the source entity, as identified in step 1   
- target entity: name of the target entity, as identified in step 1   
- relationship description: explanation as to why you think the source entity and the target entity are   
related to each other   
- relationship strength: a numeric score indicating strength of the relationship between the source entity   
and target entity

Format each relationship as ("relationship"{tuple delimiter}<source entity>{tuple delimiter}<target entity>{tuple delimiter}<relationship description>{tuple delimiter}<relationship strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use $\pm \ast \{ \mathrm { r e c o r d \_ d e l i m i t e r } \} \ast \ast$ as the list delimiter.

4. When finished, output {completion delimiter} ---Examples---

Entity types: ORGANIZATION,PERSON

Input:

The Fed is scheduled to meet on Tuesday and Wednesday, with the central bank planning to release its latest policy decision on Wednesday at $2 : 0 0 \ \mathbf { p } . \mathtt { m }$ . ET, followed by a press conference where Fed Chair Jerome Powell will take questions. Investors expect the Federal Open Market Committee to hold its benchmark interest rate steady in a range of 5.25%-5.5%.

Output:

("entity"{tuple delimiter}FED{tuple delimiter}ORGANIZATION{tuple delimiter}The Fed is the Federal Reserve,   
which is setting interest rates on Tuesday and Wednesday)   
{record delimiter}   
("entity"{tuple delimiter}JEROME POWELL{tuple delimiter}PERSON{tuple delimiter}Jerome Powell is the chair   
of the Federal Reserve)   
{record delimiter}   
("entity"{tuple delimiter}FEDERAL OPEN MARKET COMMITTEE{tuple delimiter}ORGANIZATION{tuple delimiter}The   
Federal Reserve committee makes key decisions about interest rates and the growth of the United States   
money supply)   
{record delimiter}   
("relationship"{tuple delimiter}JEROME POWELL{tuple delimiter}FED{tuple delimiter}Jerome Powell is the   
Chair of the Federal Reserve and will answer questions at a press conference{tuple delimiter}9)   
{completion delimiter}

...More examples...

---Real Data---

Entity types: {entity types}   
Input:   
{input text}

Output:

# E.2 Community Summary Generation

---Role---   
You are an AI assistant that helps a human analyst to perform general information discovery. Information discovery is the process of identifying and assessing relevant information associated with certain entities (e.g., organizations and individuals) within a network.

---Goal---

Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact. The content of this report includes an overview of the community’s key entities, their legal compliance, technical capabilities, reputation, and noteworthy claims.

---Report Structure---

The report should include the following sections:

- TITLE: community’s name that represents its key entities - title should be short but specific. When possible, include representative named entities in the title.   
- SUMMARY: An executive summary of the community’s overall structure, how its entities are related to each other, and significant information associated with its entities.   
- IMPACT SEVERITY RATING: a float score between $0 { - } 1 0$ that represents the severity of IMPACT posed by entities within the community. IMPACT is the scored importance of a community.   
- RATING EXPLANATION: Give a single sentence explanation of the IMPACT severity rating.   
- DETAILED FINDINGS: A list of 5-10 key insights about the community. Each insight should have a short summary followed by multiple paragraphs of explanatory text grounded according to the grounding rules below. Be comprehensive.   
Return output as a well-formed JSON-formatted string with the following format:   
{{   
"title": <report title>,   
"summary": <executive summary>,   
"rating": <impact severity rating>,   
"rating explanation": <rating explanation>,   
"findings": [   
{{   
"summary":<insight 1 summary>,   
"explanation": <insight 1 explanation>   
}},   
{{   
"summary":<insight 2 summary>,   
"explanation": <insight 2 explanation>   
}}   
]   
}}   
---Grounding Rules---

Points supported by data should list their data references as follows:

"This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); <dataset name> (record ids)]."

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (1), Entities (5, 7); Relationships (23); Claims (7, 2, 34, 64, 46, +more)]."

where 1, 5, 7, 23, 2, 34, 46, and 64 represent the id (not the index) of the relevant data record.

o not include information where the supporting evidence for it is not provide

---Example---

Input:

Entities

id,entity,description 5,VERDANT OASIS PLAZA,Verdant Oasis Plaza is the location of the Unity March 6,HARMONY ASSEMBLY,Harmony Assembly is an organization that is holding a march at Verdant Oasis Plaza

Relationships

id,source,target,description   
37,VERDANT OASIS PLAZA,UNITY MARCH,Verdant Oasis Plaza is the location of the Unity March   
38,VERDANT OASIS PLAZA,HARMONY ASSEMBLY,Harmony Assembly is holding a march at Verdant Oasis Plaza   
39,VERDANT OASIS PLAZA,UNITY MARCH,The Unity March is taking place at Verdant Oasis Plaza   
40,VERDANT OASIS PLAZA,TRIBUNE SPOTLIGHT,Tribune Spotlight is reporting on the Unity march taking place at   
Verdant Oasis Plaza   
41,VERDANT OASIS PLAZA,BAILEY ASADI,Bailey Asadi is speaking at Verdant Oasis Plaza about the march   
43,HARMONY ASSEMBLY,UNITY MARCH,Harmony Assembly is organizing the Unity March

Output:

{{   
"title": "Verdant Oasis Plaza and Unity March",   
"summary": "The community revolves around the Verdant Oasis Plaza, which is the location of the Unity   
March. The plaza has relationships with the Harmony Assembly, Unity March, and Tribune Spotlight, all of   
which are associated with the march event.",   
"rating": 5.0,   
"rating explanation": "The impact severity rating is moderate due to the potential for unrest or conflict   
during the Unity March.",   
"findings": [   
{{   
"summary": "Verdant Oasis Plaza as the central location",   
"explanation": "Verdant Oasis Plaza is the central entity in this community, serving as the location for   
the Unity March. This plaza is the common link between all other entities, suggesting its significance   
in the community. The plaza’s association with the march could potentially lead to issues such as   
public disorder or conflict, depending on the nature of the march and the reactions it provokes. [Data:   
Entities (5), Relationships (37, 38, 39, 40, 41,+more)]"   
}},   
{{   
"summary": "Harmony Assembly’s role in the community",   
"explanation": "Harmony Assembly is another key entity in this community, being the organizer of the   
march at Verdant Oasis Plaza. The nature of Harmony Assembly and its march could be a potential source of   
threat, depending on their objectives and the reactions they provoke. The relationship between Harmony   
Assembly and the plaza is crucial in understanding the dynamics of this community. [Data: Entities(6),   
Relationships (38, 43)]"   
}},   
{{   
"summary": "Unity March as a significant event",   
"explanation": "The Unity March is a significant event taking place at Verdant Oasis Plaza. This event   
is a key factor in the community’s dynamics and could be a potential source of threat, depending on the   
nature of the march and the reactions it provokes. The relationship between the march and the plaza is   
crucial in understanding the dynamics of this community. [Data: Relationships (39)]"   
}},   
{{   
"summary": "Role of Tribune Spotlight", "explanation": "Tribune Spotlight is reporting on the Unity   
March taking place in Verdant Oasis Plaza. This suggests that the event has attracted media attention,   
which could amplify its impact on the community. The role of Tribune Spotlight could be significant in   
shaping public perception of the event and the entities involved. [Data: Relationships (40)]"   
}}   
]   
}}   
---Real Data---

Use the following text for your answer. Do not make anything up in your answer.

Input:

{input text} ...Report Structure and Grounding Rules Repeated...

Output:

# E.3 Community Answer Generation

---Role---

You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.

---Goal---

Generate a response of the target length and format that responds to the user’s question, summarize all the reports from multiple analysts who focused on different parts of the dataset, and incorporate any relevant general knowledge.

Note that the analysts’ reports provided below are ranked in the \*\*descending order of helpfulness\*\*.

If you don’t know the answer, just say so. Do not make anything up.

The final response should remove all irrelevant information from the analysts’ reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

The response should also preserve all the data references previously included in the analysts’ reports, but do not mention the roles of multiple analysts in the analysis process.

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"   
where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record. Do not include information where the supporting evidence for it is not provided.   
---Target response length and format---   
{response type}   
---Analyst Reports---   
{report data}

...Goal and Target response length and format repeated...

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

Output:

# E.4 Global Answer Generation

---Role---   
You are a helpful assistant responding to questions about data in the tables provided. ---Goal---

Generate a response of the target length and format that responds to the user’s question, summarize all relevant information in the input data tables appropriate for the response length and format, and incorporate any relevant general knowledge.

If you don’t know the answer, just say so. Do not make anything up.

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:

"This is an example sentence supported by data references [Data: Reports (report ids)]"

ote: the prompts for SS (semantic search) and TS (text summarization) conditions use ”Sources” in place of ”Reports” above.

Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

For example:

"Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

Do not include information where the supporting evidence for it is not provided.

At the beginning of your response, generate an integer score between 0-100 that indicates how \*\*helpful\*\* is this response in answering the user’s question. Return the score in this format: <ANSWER HELPFULNESS> score value </ANSWER HELPFULNESS>.   
---Target response length and format---   
{response type}   
---Data tables---   
{context data}   
...Goal and Target response length and format repeated...

Output:

# F Evaluation Prompts

# F.1 Relative Assessment Prompt

---Role---

You are a helpful assistant responsible for grading two answers to a question that are provided by two different people.

---Goal---

Given a question and two answers (Answer 1 and Answer 2), assess which answer is better according to the following measure:

{criteria}

Your assessment should include two parts:   
- Winner: either 1 (if Answer 1 is better) and 2 (if Answer 2 is better) or 0 if they are fundamentally similar and the differences are immaterial.   
- Reasoning: a short explanation of why you chose the winner with respect to the measure described above.   
Format your response as a JSON object with the following structure:   
{{   
"winner": <1, 2, or 0>,   
"reasoning": "Answer 1 is better because <your reasoning>."   
}}   
---Question---   
{question}   
---Answer 1---   
{answer1}   
---Answer 2---   
{answer2}

Assess which answer is better according to the following measure: {criteria}

Output:

# F.2 Relative Assessment Metrics

CRITERIA $\begin{array} { r l r } { \mathrm { ~ } } & { { } = } & { \left\{ \begin{array} { r l } \end{array} \right. } \end{array}$

"comprehensiveness": "How much detail does the answer provide to cover all the aspects and details of the question? A comprehensive answer should be thorough and complete, without being redundant or irrelevant. For example, if the question is ’What are the benefits and drawbacks of nuclear energy?’, a comprehensive answer would provide both the positive and negative aspects of nuclear energy, such as its efficiency, environmental impact, safety, cost, etc. A comprehensive answer should not leave out any important points or provide irrelevant information. For example, an incomplete answer would only provide the benefits of nuclear energy without describing the drawbacks, or a redundant answer would repeat the same information multiple times.",

"diversity": "How varied and rich is the answer in providing different perspectives and insights on the question? A diverse answer should be multi-faceted and multi-dimensional, offering different viewpoints and angles on the question. For example, if the question is ’What are the causes and effects of climate change?’, a diverse answer would provide different causes and effects of climate change, such as greenhouse gas emissions, deforestation, natural disasters, biodiversity loss, etc. A diverse answer should also provide different sources and evidence to support the answer. For example, a single-source answer would only cite one source or evidence, or a biased answer would only provide one perspective or opinion.",

"directness": "How specifically and clearly does the answer address the question? A direct answer should provide a clear and concise answer to the question. For example, if the question is ’What is the capital of France?’, a direct answer would be ’Paris’. A direct answer should not provide any irrelevant or unnecessary information that does not answer the question. For example, an indirect answer would be ’The capital of France is located on the river Seine’.",

"empowerment": "How well does the answer help the reader understand and make informed judgements about   
the topic without being misled or making fallacious assumptions. Evaluate each answer on the quality of   
answer as it relates to clearly explaining and providing reasoning and sources behind the claims in the   
answer."

# G Statistical Analysis

Table 6: Pairwise comparisons of six conditions on four metrics across 125 questions and two datasets. For each question and metric, the winning condition received a score of 100, the losing condition received a score of 0, and in the event of a tie, each condition was scored 50. These scores were then averaged over five evaluation runs for each condition. Results of Shapiro-Wilk tests indicated that the data did not follow a normal distribution. Thus, non-parametric tests (Wilcoxon signed-rank tests) were employed to assess the performance differences between pairs of conditions, with Holm-Bonferroni correction applied to account for multiple pairwise comparisons. The corrected $\mathsf { p }$ -values that indicated statistically significant differences are highlighted in bold.

<table><tr><td colspan="9">Podcast Transcripts</td><td colspan="4">News Articles</td></tr><tr><td colspan="3">Condition 1</td><td>Condition 2 Mean 1</td><td>Mean2</td><td>Z-value</td><td>p-value</td><td>Mean1</td><td></td><td>Mean2</td><td>Z-value</td><td>p-value</td></tr><tr><td rowspan="15"></td><td>CO</td><td>TS</td><td>50.24</td><td>49.76</td><td>-0.06</td><td>1</td><td></td><td>55.52</td><td>44.48</td><td>-2.03</td><td>0.17</td></tr><tr><td>C1</td><td>TS</td><td>51.92</td><td>48.08</td><td>-1.56</td><td>0.633</td><td>58.8</td><td>41.2</td><td></td><td>-3.62</td><td>0.002</td></tr><tr><td>C2</td><td>TS</td><td>57.28</td><td>42.72</td><td>-4.1</td><td>&lt;0.001</td><td>62.08</td><td></td><td>37.92</td><td>-5.07</td><td>&lt;0.001</td></tr><tr><td>C3</td><td>TS</td><td>56.48</td><td>43.52</td><td>-3.42</td><td>0.006</td><td>63.6</td><td></td><td>36.4</td><td>-5.63</td><td>&lt;0.001</td></tr><tr><td>CO</td><td>SS</td><td>71.92</td><td>28.08</td><td>-6.2</td><td>&lt;0.001</td><td></td><td>71.76</td><td>28.24</td><td>-6.3</td><td>&lt;0.001</td></tr><tr><td>C1</td><td>SS</td><td>75.44</td><td>24.56</td><td>-7.45</td><td>&lt;0.001</td><td></td><td>74.72</td><td>25.28</td><td>-7.78</td><td>&lt;0.001</td></tr><tr><td>C2</td><td>SS</td><td>77.76</td><td>22.24</td><td>-8.17</td><td></td><td>&lt;0.001</td><td>79.2</td><td>20.8</td><td>-8.34</td><td>&lt;0.001</td></tr><tr><td>Comprehensiveness C3</td><td>SS</td><td>78.96</td><td>21.04</td><td>-8.12</td><td></td><td>&lt;0.001</td><td>79.44</td><td>20.56</td><td>-8.44</td><td>&lt;0.001</td></tr><tr><td>TS</td><td>SS</td><td>83.12</td><td>16.88</td><td>-8.85</td><td></td><td>&lt;0.001</td><td>79.6</td><td>20.4</td><td>-8.27</td><td>&lt;0.001</td></tr><tr><td>CO</td><td>C1</td><td>53.2</td><td>46.8</td><td></td><td>-1.96</td><td>0.389</td><td>51.92</td><td>48.08</td><td>-0.45</td><td>0.777</td></tr><tr><td>C0</td><td>C2</td><td></td><td>50.24</td><td>49.76</td><td>-0.23</td><td>1</td><td>53.68</td><td>46.32</td><td>-1.54</td><td>0.371</td></tr><tr><td>C1</td><td>C2</td><td></td><td>51.52</td><td>48.48</td><td>-1.62</td><td>0.633</td><td>57.76</td><td>42.24</td><td>-4.01</td><td>&lt;0.001</td></tr><tr><td>C0 C1</td><td>C3</td><td>49.12</td><td>50.88</td><td></td><td>-0.56 1</td><td></td><td>52.16</td><td>47.84</td><td>-0.86</td><td>0.777</td></tr><tr><td>C2</td><td>C3</td><td>50.32</td><td>49.68</td><td>-0.66</td><td>1</td><td></td><td>55.12</td><td>44.88</td><td>-2.94</td><td>0.016</td></tr><tr><td></td><td></td><td>C3</td><td>52.24 50.24</td><td>47.76 49.76</td><td>-1.97</td><td>0.389</td><td>58.64</td><td>41.36</td><td>-3.68</td><td>0.002</td></tr><tr><td rowspan="15">Diversity</td><td>C0 C1</td><td>TS</td><td>50.48</td><td>49.52</td><td>-0.11 -0.12</td><td>1 1</td><td>46.88 54.64</td><td>53.12</td><td>-1.38</td><td>0.676</td></tr><tr><td>C2</td><td>TS</td><td></td><td></td><td></td><td></td><td></td><td>45.36</td><td>-1.88</td><td>0.298</td></tr><tr><td></td><td>TS</td><td>57.12</td><td>42.88</td><td>-2.84</td><td>0.036</td><td>55.76</td><td>44.24</td><td>-2.16</td><td>0.184</td></tr><tr><td>C3</td><td>TS</td><td>54.32</td><td>45.68</td><td>-2.39</td><td>0.1</td><td>60.16</td><td>39.84</td><td>-4.07</td><td>&lt;0.001</td></tr><tr><td>CO</td><td>SS</td><td>76.56</td><td>23.44</td><td>-7.12</td><td>&lt;0.001</td><td>62.08</td><td>37.92</td><td>-3.57</td><td>0.003</td></tr><tr><td>C1</td><td>SS</td><td>75.44</td><td>24.56</td><td>-7.33</td><td>&lt;0.001</td><td>64.96</td><td>35.04</td><td>-4.92</td><td>&lt;0.001</td></tr><tr><td>C2</td><td>SS</td><td>80.56</td><td>19.44</td><td>-8.21</td><td>&lt;0.001</td><td>70.56</td><td>29.44</td><td>-6.29</td><td>&lt;0.001</td></tr><tr><td>C3</td><td>SS</td><td>80.8</td><td>19.2</td><td>-8.3</td><td>&lt;0.001</td><td>69.12</td><td>30.88</td><td>-5.53</td><td>&lt;0.001</td></tr><tr><td>TS</td><td>Ss</td><td>82.08</td><td>17.92</td><td>-8.43</td><td>&lt;0.001</td><td>67.2</td><td>32.8</td><td>-4.85</td><td>&lt;0.001</td></tr><tr><td>CO</td><td>C1</td><td>49.76</td><td>50.24</td><td>-0.13</td><td>1</td><td>39.68</td><td>60.32</td><td>-3.61</td><td>0.003</td></tr><tr><td>C0</td><td>C2</td><td>46.32</td><td>53.68</td><td>-1.5</td><td>0.669</td><td>40.96</td><td>59.04</td><td>-3.14</td><td>0.012</td></tr><tr><td>C1</td><td>C2</td><td>44.08</td><td>55.92</td><td>-3.27</td><td>0.011</td><td>50.24</td><td>49.76</td><td>-0.22</td><td>1</td></tr><tr><td>C0</td><td>C3</td><td>44</td><td>56</td><td>-2.6</td><td>0.065</td><td>41.04</td><td>58.96</td><td>-3.47</td><td>0.004</td></tr><tr><td>C1 C2</td><td>C3</td><td>45.44</td><td>54.56 51.52</td><td>-2.98</td><td>0.026</td><td>49.52</td><td>50.48</td><td>-0.01</td><td>1</td></tr><tr><td>CO</td><td>C3</td><td>48.48 40.96</td><td>59.04</td><td>-0.96 -4.3</td><td>1 &lt;0.001</td><td>50.96 42.24</td><td>49.04 57.76</td><td>-0.39 -3.32</td><td>1 0.012</td></tr><tr><td rowspan="15">Empowerment</td><td>C1</td><td>TS TS</td><td>45.2</td><td>54.8</td><td>-3.76</td><td>0.002</td><td>50</td><td>50</td><td>-0.12</td><td>1</td></tr><tr><td>C2</td><td>TS</td><td>47.68</td><td>52.32</td><td>-2.2</td><td>0.281</td><td>49.52</td><td>50.48</td><td>-0.22</td><td>1</td></tr><tr><td>C3</td><td>TS</td><td>48.72</td><td>51.28</td><td>-1.27</td><td>1</td><td>51.68</td><td>48.32</td><td>-1.2</td><td>1</td></tr><tr><td>CO</td><td>SS</td><td>42.96</td><td>57.04</td><td>-3.71</td><td>0.003</td><td>42.72</td><td>57.28</td><td>-3.12</td><td></td></tr><tr><td>C1</td><td>ss</td><td>47.68</td><td>52.32</td><td>-1.5</td><td>0.936</td><td>51.36</td><td>48.64</td><td></td><td>0.022</td></tr><tr><td>C2</td><td></td><td>50.72</td><td>49.28</td><td></td><td></td><td></td><td></td><td>-0.84</td><td>1</td></tr><tr><td>C3</td><td>SS</td><td>48.96</td><td>51.04</td><td>-0.55 -0.57</td><td>1 1</td><td>49.84 49.52</td><td>50.16 50.48</td><td>-0.2 -0.08</td><td>1 1</td></tr><tr><td>TS</td><td>Ss SS</td><td>57.52</td></table>