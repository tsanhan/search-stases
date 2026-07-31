import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatAnthropic } from "@langchain/anthropic";
import {
  AIMessage,
  BaseMessage,
  HumanMessage,
  MessageStructure,
  MessageToolSet,
  MessageType,
} from "@langchain/core/messages";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
  TypedPromptInputValues,
} from "@langchain/core/prompts";
import {
  StateGraph,
  Annotation,
  StateDefinition,
  SingleReducer,
} from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  DynamicStructuredTool,
  DynamicTool,
  tool,
} from "@langchain/core/tools";
import { MongoDBSaver } from "@langchain/langgraph-checkpoint-mongodb";
import { MongoClient } from "mongodb";
import {
  MongoDBAtlasVectorSearch,
  MongoDBAtlasVectorSearchLibArgs,
} from "@langchain/mongodb";
import { z } from "zod";
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

  const stasLookupTool: DynamicStructuredTool = tool(
    async ({ query, n = 4 }: { query: string; n?: number }) => {
      console.log("[stas lookup] tool called with query:", query);
      console.log("[stas lookup] requested result count:", n);

      const dbConfig = {
        collection: collection,
        indexName: "stas_vector_index",
        testKey: "embedding_text",
        embeddingKey: "embedding",
      } as unknown as MongoDBAtlasVectorSearchLibArgs;

      const embeddingInstance: EmbeddingsInterface = new OpenAIEmbeddings(); // using the same embedding mechanism (from OpenAI)

      const vectorStore = new MongoDBAtlasVectorSearch(
        embeddingInstance,
        dbConfig,
      );

      console.log("[stas lookup] executing vector search against MongoDB Atlas...");
      const result = await vectorStore.similaritySearchWithScore(query, n);
      console.log("[stas lookup] db response:", JSON.stringify(result, null, 2));

      return JSON.stringify(result);
    },
    {
      name: "lookup_for_a_stas",
      description:
        "Gathers details about different Stas's details from the Stases Database",
      schema: z.object({
        query: z.string().describe("the search query"),
        n: z
          .number()
          .optional()
          .default(4)
          .describe("Number of results to return"),
      }),
    },
  );

  const tools = [stasLookupTool];

  // extract the state typing via `GraphState.State`
  const toolNode = new ToolNode<typeof graphState.State>(tools);

  const model = new ChatAnthropic({
    model: "claude-haiku-4-5",
    temperature: 0,
  }).bindTools(tools);

  async function callModel(state: typeof graphState.State) {
    const prompt: ChatPromptTemplate = ChatPromptTemplate.fromMessages([
      [
        "system",
        `You are a helpful AI assistant, collaborating with other assistants. 
        Use the provided tools to progress towards answering the question. 
        If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off. 
        Execute what you can to make progress. 
        If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop. 
        You have access to the following tools:

        {tool_names}.\n
        {system_message}\n
        
        Current time: {time}.`,
      ],
      new MessagesPlaceholder("messages"),
    ]);

    const formattedPrompt: BaseMessage<
      MessageStructure<MessageToolSet>,
      MessageType
    >[] = await prompt.formatMessages({
      system_message:
        "You are a helpful Chatbot agent helping query different Stas related data",
      time: new Date().toISOString(),
      tool_names: tools.map((x) => x.name).join(", "),
      messages: state.messages ?? [],
    });

    const result = await model.invoke(formattedPrompt);

    return { messages: [result] };
  }

  function shouldContinue(state: typeof graphState.State) {
    const messages = state.messages as BaseMessage[];
    const lastMessage = messages[messages.length - 1] as AIMessage;

    // if the llm makes a tool call, the we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }

    // otherwise we stop
    return "__end__";
  }

  const workflow = new StateGraph(graphState)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge("tools", "agent")
    .addEdge("__start__", "agent")
    .addConditionalEdges("agent", shouldContinue);

  const checkpointer = new MongoDBSaver({ client, dbName });

  const app = workflow.compile({ checkpointer });

  const finalState = (await app.invoke(
    { messages: [new HumanMessage(query)] },
    { recursionLimit: 15, configurable: { thread_id: threadId } },
  )) as typeof graphState.State & { messages: BaseMessage[] };

  console.log(
    (finalState.messages[finalState.messages.length - 1] as AIMessage).content,
  );

  return (finalState.messages[finalState.messages.length - 1] as AIMessage).content
}
