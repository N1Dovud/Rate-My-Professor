import { NextResponse } from 'next/server';
import { Pinecone } from '@pinecone-database/pinecone';
import { OpenAI } from 'openai';
import fetch from "node-fetch";

const systemPrompt = `
You are a rate my professor agent to help students find classes, that takes in user questions and answers them.
For every user question, the top 3 professors that match the user question are returned.
The user can then ask for more information about a specific professor.
`;
export async function POST(req) {
    const data = await req.json();
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY
    });
    const index = pc.index('rag').namespace('ns1');
    const openai = new OpenAI({
        baseURL: "https://openrouter.ai/api/v1",
        apiKey: process.env.OPENROUTER_API_KEY,
    });
    const text = data[data.length - 1].content;

    const model_id = "sentence-transformers/all-MiniLM-L6-v2";
    const hf_token = process.env.HF_TOKEN;

    const api_url = `https://api-inference.huggingface.co/pipeline/feature-extraction/${model_id}`;
    const headers = { 
        Authorization: `Bearer ${hf_token}`,
        "Content-Type": "application/json",
     };

    const getEmbedding = async texts => {
        const response = await fetch(api_url,{
            method: "POST",
            headers: headers,
            body: JSON.stringify({
                inputs: texts,
            })
        }
        );
        return response.json();    
    }
    const embedding = await getEmbedding(text);
    const results = await index.query({
        topK: 5,
        includeMetadata: true,
        vector: embedding
    });
    let resultString = '';
    results.matches.forEach(match => {
        resultString += `
        Returned Results: 
        Profesor: ${match.id}
        Review: ${match.metadata.review}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n`;
        }
        );
    const lastMessage = data[data.length - 1];
    const lastMessageContent = lastMessage.content + resultString;
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
    const completion = await openai.chat.completions.create({
        messages: [
            {
                role: 'system', content: systemPrompt
            },
            {
                role: 'user', content: lastMessageContent
            },
        ],
        model: 'openchat/openchat-7b:free',
    });
    return NextResponse.json({ content: completion.choices[0]?.message.content });
}