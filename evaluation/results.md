## Les collections

#### Collection 256_1000
- Nombre de chunk: 
- Temps d'ingest: 6:
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 256_950
- Nombre de chunk: 191366
- Temps d'ingest: 5:50
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions

#### Collection 512_1000
- Nombre de chunk: 
- Temps d'ingest:
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 512_950
- Nombre de chunk: 82566
- Temps d'ingest: 4:26
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions

#### Collection 768_1000
- Nombre de chunk: 
- Temps d'ingest: 
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 768_950
- Nombre de chunk: 53865
- Temps d'ingest: 3:38
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions
  
---
## Les résultats

### Les LLM seuls

#### Test 1: 
Modèle: mistral-7b-instruct-v0.2.Q8_0 en GGUF local
Prompt: aucun, uniquement la question de Q&A
Temps d'inférence: 1:00:00
Scores: 

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.1814  | 0.0425  | 0.1216  | 0.0128 | 0.0348               | 0.4667        | 0.6041     | 0.5241 |

#### Test 2: 
Modèle: gemma-7b-it de Groq
Prompt: aucun, uniquement la question de Q&A
Temps d'inférence: 1:24
Scores: 

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.1364  | 0.0437  | 0.1037  | 0.0143 | 0.0251               | 0.4811        | 0.6347     | 0.5458 |

### Les RAG

#### Test 3: 
Modèle: SimpleRAG avec gemma-7b-it de Groq
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 950_256
Temps d'inférence: 1:25
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.2455  | 0.073   | 0.1708  | 0.0388 | 0.0751               | 0.5315        | 0.5937     | 0.5599 |

#### Test 4: 
Modèle: SimpleRAG avec gemma-7b-it de Groq
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 950_512
Temps d'inférence: 1:28
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.2363  | 0.0632  | 0.1598  | 0.0335 | 0.0696               | 0.5235        | 0.5878     | 0.5525 |
