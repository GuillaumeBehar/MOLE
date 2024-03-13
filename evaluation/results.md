#### Collection 1000_256
- Nombre de chunk: 
- Temps d'ingest:
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 950_256
- Nombre de chunk: 191366
- Temps d'ingest: 5:50
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions

#### Collection 1000_512
- Nombre de chunk: 
- Temps d'ingest:
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 950_512
- Nombre de chunk: 82566
- Temps d'ingest: 4:26
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions

#### Collection 1000_768
- Nombre de chunk: 
- Temps d'ingest: 
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
#### Collection 950_768
- Nombre de chunk: 53865
- Temps d'ingest: 3:38
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions


#### Test 1: 
Modèle: mistral-7b-instruct-v0.2.Q8_0 en GGUF local
Prompt: aucun, uniquement la question de Q&A
Temps d'inférence: 1 heure
Scores: 

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.1814  | 0.0425  | 0.1216  | 0.0128 | 0.0348               | 0.4667        | 0.6041     | 0.5241 |



#### Test 2: 
Modèle: SimpleRAG avec Mistral-7B-Instruct-v0.2 de HuggingChat
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 
Temps d'inférence: 7 minutes
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.2374  | 0.0586  | 0.1622  | 0.0263 | 0.0642               | 0.5559        | 0.6204     | 0.5853 | 

#### Test 2: 
Modèle: SimpleRAG avec Mistral-7B-Instruct-v0.2 de HuggingChat
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 
- Nombre de documents: 1000
- Nombre de chunks: 190966
- Taille: 256 caractères
- Chevauchement: 64 caractères
- Temps d'ingest: ? minutes
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
Temps d'inférence: 7 minutes
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.2374  | 0.0586  | 0.1622  | 0.0263 | 0.0642               | 0.5559        | 0.6204     | 0.5853 | 


#### Test 3: 
Modèle: SimpleRAG avec mistral-7b-instruct-v0.2.Q8_0 en GGUF local
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 
- Nombre de documents: 1000
- Nombre de chunks: 190966
- Taille: 256 caractères
- Chevauchement: 64 caractères
- Temps d'ingest: ? minutes
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
Temps d'inférence: 1 heure
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.194   | 0.0402  | 0.1353  | 0.0163 | 0.0456               | 0.5229        | 0.5906     | 0.553  |


#### Test 4: 
Modèle: SimpleRAG avec mistral-7b-instruct-v0.2.Q8_0 en GGUF local
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 
- Nombre de documents: 1000
- Nombre de chunks: 82566
- Taille: 512 caractères
- Chevauchement: 64 caractères
- Temps d'ingest: 9 minutes
- La collection contient les documents spécifiés par Q&A pour répondre aux questions
Temps d'inférence: 7 minutes
Scores:

| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.194   | 0.0402  | 0.1353  | 0.0163 | 0.0456               | 0.5229        | 0.5906     | 0.553  |