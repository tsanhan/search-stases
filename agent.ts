import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import { AIMessage, BaseMessage, HumanMessage } from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  StateGraph,
  Annotation,
  StateDefinition,
  SingleReducer,
} from "@langchain/langgraph";
import { DynamicTool, tool } from "@langchain/core/tools";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoDBAtlasVectorSearch, MongoDBAtlasVectorSearchLibArgs } from "@langchain/mongodb";
import { MongoClient } from "mongodb";
import { string, z } from "zod";
import "dotenv/config";
import { Document } from "langchain";
import { Collection } from "mongodb";
import { EmbeddingsInterface } from "@langchain/core/embeddings";

export async function callAgent(
  client: MongoClient,
  query: string,
  threadId: string,
) {
  // define the mongodb database and collection
  const dbName: string = "stas_database";
  const db = client.db(dbName);
  const collection = db.collection("stas_collection");

  const reducer: SingleReducer<BaseMessage[], BaseMessage | BaseMessage[]> = {
    reducer: (left: BaseMessage[], right: BaseMessage | BaseMessage[]) =>
      left.concat(right),
  };
  const messages = Annotation<BaseMessage[]>(reducer);
  const stateDef: StateDefinition = { messages };
  const graphState = Annotation.Root(stateDef);

  const stasLookupTool: DynamicTool = tool(
    async ({ query, n = 10 }: { query: string; n?: number }) => {
        console.log("stas lookup tool called");

        const dbConfig = {
          collection: collection,
          indexName: "stas_vector_index",
          testKey: "embedding_text",
          embeddingKey: "embedding",
        } as unknown as MongoDBAtlasVectorSearchLibArgs;

        const embeddingInstance: EmbeddingsInterface = new OpenAIEmbeddings(); // using the sage embedding mechanism (from OpenAI)

        const vectorStore = new MongoDBAtlasVectorSearch(
          embeddingInstance,
          dbConfig
        );

        const result = await vectorStore.similaritySearchWithScore(query, n);

        return JSON.stringify(result);
    },
    {
        name: "lookup_for_a_stas",
        description: "Gathers details about different Stas's details from the Stases Database",
        schema: z.object({
            query: z.string().describe("the search query"),
            n: z.number().optional().default(10).describe("Number of results to return")
        })
    }
  )
}
