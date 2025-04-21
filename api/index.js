import { Pinecone } from '@pinecone-database/pinecone';
import OpenAI from 'openai';

// Initialize clients - these are initialized once and reused across function invocations
let openai;
let pineconeIndex;
let pineconeInitialized = false;

const initClients = async () => {
  // Only initialize once
  if (!openai) {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  }
  
  if (!pineconeInitialized) {
    try {
      const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
      const indexName = process.env.PINECONE_INDEX_NAME || 'text-embeddings';
      
      const indexList = await pinecone.listIndexes();
      const existingIndexes = indexList.indexes?.map(i => i.name) || [];

      if (!existingIndexes.includes(indexName)) {
        console.log(`Creating new Pinecone index: ${indexName}`);
        await pinecone.createIndex({
          name: indexName,
          dimension: 1536,
          metric: 'cosine',
          spec: {
            serverless: {
              cloud: process.env.PINECONE_CLOUD || 'aws',
              region: process.env.PINECONE_REGION || 'us-east-1'
            }
          }
        });

        // Wait for index initialization
        await new Promise(resolve => setTimeout(resolve, 30000));
      }

      pineconeIndex = pinecone.Index(indexName);
      pineconeInitialized = true;
      console.log('Pinecone index ready');
    } catch (error) {
      console.error('Pinecone initialization error:', error);
      throw error;
    }
  }

  return { openai, pineconeIndex };
};

// Enhanced CORS headers helper function - always allows localhost during development
const setCorsHeaders = (res) => {
  // Default to allowing all origins during development
  const allowedOrigin = process.env.FRONTEND_URL || '*';
  
  // Check if we need to explicitly allow localhost development servers
  const origin = res.req.headers.origin;
  if (origin && (
    origin.includes('localhost') || 
    origin.includes('127.0.0.1') || 
    allowedOrigin === '*'
  )) {
    res.setHeader('Access-Control-Allow-Origin', origin);
  } else if (allowedOrigin !== '*') {
    res.setHeader('Access-Control-Allow-Origin', allowedOrigin);
  } else {
    // Fallback to '*' if no specific origin is matched
    res.setHeader('Access-Control-Allow-Origin', '*');
  }
  
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader('Access-Control-Allow-Headers', 
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
  );
};

// Health check handler
const handleHealth = (req, res) => {
  return res.status(200).json({ status: 'ok' });
};

// Embeddings handler
const handleEmbeddings = async (req, res) => {
  try {
    // Initialize clients
    const { openai, pineconeIndex } = await initClients();
    
    const { text, chunkId } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: "Text content is required" });
    }
    
    console.log(`Processing text chunk (${text.length} chars)${chunkId ? ` with ID ${chunkId}` : ''}`);
    
    // Generate OpenAI embedding
    const embedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: text,
    });
    
    // Generate a unique ID or use the provided chunkId
    const id = chunkId || Date.now().toString();
    
    // Store in Pinecone
    const vectorData = {
      id: id,
      values: embedding.data[0].embedding,
      metadata: { text }
    };
    
    await pineconeIndex.upsert([vectorData]);
    console.log(`Successfully stored embedding with ID: ${id}`);
    
    return res.status(201).json({
      success: true,
      id: id,
      text: text.length > 100 ? text.substring(0, 100) + '...' : text
    });
    
  } catch (error) {
    console.error('Error in /api/embeddings:', error);
    return res.status(500).json({ error: error.message });
  }
};

// Ask handler
const handleAsk = async (req, res) => {
  try {
    // Initialize clients
    const { openai, pineconeIndex } = await initClients();
    
    const { question, documentID } = req.body;

    console.log(`Received question: "${question}" for document ID: ${documentID}`);

    if (!question || typeof question !== 'string') {
      return res.status(400).json({ error: "Valid question string is required" });
    }

    if (!documentID) {
      return res.status(400).json({ error: "Document ID is required" });
    }

    // Generate question embedding
    const questionEmbedding = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: question,
    });

    console.log("Generated embedding for question");

    // Query Pinecone for similar vectors
    const queryResult = await pineconeIndex.query({
      vector: questionEmbedding.data[0].embedding,
      topK: 5, // Get more context
      includeMetadata: true,
    });

    console.log(`Query results: ${queryResult.matches.length} matches`);
    
    // List all available documents for debugging
    if (queryResult.matches.length > 0) {
      console.log("Available documents:");
      queryResult.matches.forEach((match, i) => {
        console.log(`${i+1}. ID: ${match.id}, Score: ${match.score}`);
      });
      
      // Try to find the exact document ID
      const exactMatch = queryResult.matches.find(match => match.id === documentID);
      if (exactMatch) {
        console.log(`Found exact match for document ID: ${documentID}`);
        
        // Generate answer using OpenAI
        const completion = await openai.chat.completions.create({
          model: process.env.OPENAI_MODEL || "gpt-3.5-turbo",
          messages: [{
            role: "system",
            content: `You are a precise, context-driven assistant. 
            - You must answer using only the provided "Context" block—do not draw on any outside knowledge. 
            - If the user's question uses different wording than the context, mentally paraphrase or expand synonyms to find the matching passage. 
            - If the answer cannot be found in the context, reply: "I'm sorry, I don't know." 
            - Keep your answer as concise as possible.
            
            Context: ${exactMatch.metadata.text}`
          }, {
            role: "user",
            content: question
          }],
          temperature: 0.7,
          max_tokens: 500
        });

        return res.json({
          question,
          answer: completion.choices[0].message.content
        });
      } else {
        console.log(`Document ID ${documentID} not found in available documents`);
        
        // Try the best match regardless of ID
        const bestMatch = queryResult.matches[0];
        if (bestMatch.score >= 0.5) { // Only use if it's a good match
          console.log(`Using best available match (score: ${bestMatch.score})`);
          
          const completion = await openai.chat.completions.create({
            model: process.env.OPENAI_MODEL || "gpt-3.5-turbo",
            messages: [{
              role: "system",
              content: `Answer the question based on the provided context. Keep your answer concise.
              
              Context: ${bestMatch.metadata.text}`
            }, {
              role: "user",
              content: question
            }],
            temperature: 0.7,
            max_tokens: 500
          });

          return res.json({
            question,
            answer: completion.choices[0].message.content
          });
        }
      }
    }

    // If we reach here, no suitable matches were found
    return res.status(404).json({ 
      error: "No relevant context found",
      question
    });

  } catch (error) {
    console.error('Error in /api/ask:', error);
    return res.status(500).json({ 
      error: "Failed to process question",
      details: error.message 
    });
  }
};

// Main API handler with routing - with improved CORS and OPTIONS handling
export default async function handler(req, res) {
  // Set CORS headers for all requests
  setCorsHeaders(res);

  // Handle OPTIONS request for CORS preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Extract path from the URL
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  const path = url.pathname.replace(/^\/api/, '') || '/';
  
  console.log(`Processing request for path: ${path} from origin: ${req.headers.origin || 'unknown'}`);

  // Route based on path
  switch (path) {
    case '/':
    case '/health':
      return handleHealth(req, res);
    
    case '/embeddings':
      if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
      }
      return handleEmbeddings(req, res);
    
    case '/ask':
      if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
      }
      return handleAsk(req, res);
    
    default:
      return res.status(404).json({ error: 'Not found' });
  }
}