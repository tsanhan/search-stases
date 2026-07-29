import { MongoClient } from "mongodb";
import "dotenv/config";
import process = require("process");

const client = new MongoClient(process.env.MONGO_ATLAS_URI as string);

async function startSrver() {
    try {
        await client.connect();
        await client.db("admin").command({ ping: 1 });
        console.log("Pinged your deployment. You successfully connected to MongoDB!");

    } catch (error) {
        console.error("Error connecting to MongoDB:", error);
    }
}
