import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoClient } from "mongodb";
import {
  MongoDBAtlasVectorSearch,
  type MongoDBAtlasVectorSearchLibArgs,
} from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";

const client = new MongoClient(process.env.MONGO_ATLAS_URI as string);

const llm = new ChatOpenAI({
  modelName: "gpt-4o-mini",
  temperature: 0,
});

const StasSchema = z.object({
  stas_id: z.string(),
  first_name: z.string(),
  last_name: z.string(),
  date_of_birth: z.string(),
  address: z.object({
    street: z.string(),
    city: z.string(),
  }),
  contact_details: z.object({
    email: z.string(),
    phone: z.string(),
  }),
  job_details: z.object({
    title: z.string(),
    department: z.string(),
  }),
  work_location: z.object({
    city: z.string(),
    street: z.string(),
    number: z.string(),
  }),
  reporting_manager: z.string().nullable(),
  skills: z.array(z.string()),
  performance_reviews: z.array(
    z.object({
      review_date: z.string(),
      rating: z.number().min(1).max(5),
      comments: z.string(),
    }),
  ),
  notes: z.string(),
});

type Stas = z.infer<typeof StasSchema>;

const parser = StructuredOutputParser.fromZodSchema(z.array(StasSchema));

async function generateSyntheticStases() {
  const prompt = `
    I have a friend in the workplace and His name is Stas.
    He is a gifted developer, Russian speaker, husband and father of two children.
    I want the next synthetic employee to be in the light of Stas, you can chage the names a bit to be different variations of "Stas", and so the job roles, and other factors.  
    Generate 10 synthetic employee records in JSON format. Each record should include the following fields: stas_id, first_name, last_name, date_of_birth, address (with street and city), contact_details (with email and phone), job_details (with title and department), work_location (with city and country), reporting_manager (nullable), skills (array of strings), performance_reviews (array of objects with review_date, rating, and comments), and notes. Ensure that the data is realistic and diverse.
    All the locations and data should be relevant to workers in Israel.


    ${parser.getFormatInstructions()}`;

  console.log("generating synthetic employee records...");

  const response = await llm.invoke(prompt);

  return parser.parse(response.content as string);
}

function createStasSummery(stas: Stas) {
  const stasDetails = `${stas.first_name} ${stas.last_name} works as a ${stas.job_details.title} in the ${stas.job_details.department} department.`;
  const skills = stas.skills.join(", ");
  const performanceReviews = stas.performance_reviews
    .map(
      (review) =>
        `On ${review.review_date}, received a rating of ${review.rating} with comments: "${review.comments}"`,
    )
    .join(" | ");
  const basicInfo = `Born on ${stas.date_of_birth}, lives at ${stas.address.street}, ${stas.address.city}. Contact: Email - ${stas.contact_details.email}, Phone - ${stas.contact_details.phone}.`;
  const workLocation = `Works in ${stas.work_location.city}, ${stas.work_location.street} ${stas.work_location.number}.`;
  const notes = `Additional notes: ${stas.notes}`;

  const summary = `${stasDetails} Skills: ${skills}. Performance Reviews: ${performanceReviews}. ${basicInfo} ${workLocation} ${notes}`;
  return summary;
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log(
      "Pinged your deployment. You successfully connected to MongoDB!",
    );

    const db = client.db("stas_database");
    const collection = db.collection(
      "stas_collection",
    ) as unknown as MongoDBAtlasVectorSearchLibArgs["collection"];
    await collection.deleteMany({}); // Clear existing data

    const syntheticStases = await generateSyntheticStases();
    console.log("Inserting synthetic employee records into the database...");

    const stasWithSummaries = await Promise.all(
      syntheticStases.map((stas: Stas) => ({
        pageContent: createStasSummery(stas),
        metadata: { ...stas },
      })),
    );

    for (const stas of stasWithSummaries) {
      await MongoDBAtlasVectorSearch.fromDocuments(
        [stas],
        new OpenAIEmbeddings(),
        {
          collection,
          indexName: "stas_vector_index",
          textKey: "pageContent",
          embeddingKey: "embedding",
        },
      );

      console.log(
        `Inserted record for ${stas.metadata.first_name} ${stas.metadata.last_name}`,
      );
    }

    console.log("Database seeding completed successfully.");
  } catch (error) {
    console.error("Error seeding the database:", error);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);
