import { MongoClient } from "mongodb";
import "dotenv/config";
import process from "process";
import express, { type Express, type Request, type Response } from "express";
import { callAgent } from "./agent";

const app: Express = express();
app.use(express.json());

const client = new MongoClient(process.env.MONGO_ATLAS_URI as string);

async function startServer() {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(
      "Pinged your deployment. You successfully connected to MongoDB!",
    );

    app.get("/", (req: Request, res: Response) => {
      res.send("Stas's LangGraph Server");
    });

    app.post("/chat", async (req: Request, res: Response) => {
      const initMessage = req.body.message;
      const threadId = Date.now().toString();
      try {
        const response = await callAgent(client, initMessage, threadId);
        res.json({ threadId, response });
      } catch (error) {
        console.log("Error starting Stas Search", error);
        res.status(500).json({ error: "Error starting Stas Search" });
      }
    });

    app.post("/chat/:threadId", async (req: Request, res: Response) => {
      const { message } = req.body;
      let { threadId } = req.params ;
      
      try {
        const response = await callAgent(client, message, threadId as string);
        res.json({ threadId, response });
      } catch (error) {
        console.log("Error starting Stas Search", error);
        res.status(500).json({ error: "Error starting Stas Search" });
      }
    });

    const PORT = process.env.PORT || 3000;

    app.listen(PORT, () => {
      console.log(`Server is on ${PORT}`);
    });
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
  }
}

startServer();
