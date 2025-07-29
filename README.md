# AI Website Chatbot Backend

This project allows you to build an AI chatbot that reads a website and answers questions about its content.

## How it works
1. It scrapes a website's text.
2. It splits that text into chunks and turns it into vector embeddings.
3. It uses GPT-4 to answer questions based on those embeddings.

## How to use
- POST to `/index` with a JSON body like `{ "url": "https://example.com" }`
- Then POST to `/query` with `{ "question": "your question here" }`

## Deployment
- Deploy this to Railway or Replit.
- Add your OpenAI API key as an environment variable: `OPENAI_API_KEY`
