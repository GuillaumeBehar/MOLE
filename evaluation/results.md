#### Test 1: 
Modèle: mistral-7b-instruct-v0.2.Q8_0, paramètres de base
Prompt: aucun, uniquement la question de Q&A
Scores: 
| Métrique          | Rouge-1 | Rouge-2 | Rouge-L | Bleu   | Avg Precision (Bleu) | Avg Precision | Avg Recall | Avg F1 |
|-------------------|---------|---------|---------|--------|----------------------|---------------|------------|--------|
| **Valeur**        | 0.1814  | 0.0425  | 0.1216  | 0.0128 | 0.0348               | 0.4667        | 0.6041     | 0.5241 |



#### Test 2: 
Modèle: RAG avec mistral-7b-instruct-v0.2.Q8_0
Prompt: {instruction}{contexte}{question} avec 5 retrieved documents
Collection: 
- Nombre de documents: 950
- Nombre de chunks: 9880
- Taille: 1024
- Chevauchement: 64
- Temps d'ingest: 6 minutes
- La collection ne contient pas les documents spécifiés par Q&A pour répondre aux questions.
Scores: