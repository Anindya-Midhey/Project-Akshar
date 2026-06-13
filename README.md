# Project Akshar
### End-to-End Intelligent Document Processing and Question Answering System

## Overview

**Project Akshar** is an end-to-end intelligent document understanding system that transforms raw document images into a searchable and explainable knowledge base. The system processes scanned and digital documents through multiple stages, including image enhancement, OCR, layout analysis, semantic retrieval, and grounded question answering.

The project combines **Computer Vision, OCR, Semantic Search, Vector Databases, and Retrieval-Augmented Generation (RAG)** to allow users to ask natural language questions about documents and receive grounded responses supported by highlighted document regions.

Unlike traditional OCR systems that only extract text, Project Akshar provides **context-aware document understanding** and **visual explainability** by linking generated answers to the exact document regions.

---

## Features

- Document image preprocessing and enhancement
- Automatic document orientation correction
- Corner detection and perspective correction
- OCR-based text extraction
- Document layout and metadata analysis
- Semantic chunking and embedding generation
- Vector database-based semantic search
- Grounded question answering using RAG
- Bounding-box-based evidence highlighting
- Explainable AI-driven document interaction
- Local LLM inference using Ollama

---

## Project Workflow

Project Akshar consists of four major modules:

### Module 1: Document Preprocessing

This module processes scanned document images to improve quality before OCR.

#### Functions
- Orientation correction
- Corner detection
- Deskewing
- Spine detection
- Double-page splitting
- Content selection
- Margin selection
- Perspective correction

**Output:** Cleaned and corrected document images.

---

### Module 2: OCR and Text Extraction

This module extracts text from processed document images.

#### Functions
- OCR processing
- Text extraction
- Text cleaning
- Structured text generation

**Output:** Machine-readable document text.

---

### Module 3: Layout and Metadata Processing

This module analyzes document structure and prepares metadata for semantic retrieval.

#### Functions
- Layout detection
- Semantic text block creation
- Page identification
- Bounding-box generation
- Metadata preparation

**Output:**
- OCR text
- Page number
- Bounding-box coordinates
- Semantic regions

---

### Module 4: Grounded RAG-Based Question Answering

This module converts processed document information into a searchable semantic knowledge base.

#### Workflow
1. Receive structured metadata from Module 3
2. Perform semantic chunking
3. Generate embeddings using the **BGE Embedding Model**
4. Store vectors in **ChromaDB**
5. Accept user query
6. Perform semantic similarity retrieval
7. Retrieve relevant chunks
8. Generate grounded answer using **Llama 3 via Ollama**
9. Highlight supporting document regions
10. Return answer with visual evidence

**Output:**
- Grounded answer
- Supporting references
- Highlighted document regions

---

## System Architecture

```text
Input Document
      ↓
Module 1 (Preprocessing)
      ↓
Module 2 (OCR & Text Extraction)
      ↓
Module 3 (Metadata & Layout Analysis)
      ↓
Module 4 (Grounded RAG Question Answering)
      ↓
Answer + Highlighted Evidence
````

---

## Tech Stack

### Programming Language

* Python

### Machine Learning / NLP

* Sentence Transformers
* BGE Embedding Model
* Llama 3
* Ollama

### OCR & Computer Vision

* OpenCV
* OCR Engine

### Vector Database

* ChromaDB

### Backend

* Flask

### Document Processing

* PyMuPDF

---

## Project Structure

```text
Project-Akshar/
│── module1/
│── module2/
│── module3/
│── module4/
│── .gitignore
│── README.md
│── project_workflow.md
│── requirements.txt
````

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Anindya-Midhey/Project-Akshar.git
cd Project-Akshar
```

### Create Conda Environment

```bash
conda create -n akshar python=3.10
conda activate akshar
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run Main Application

```bash
python main.py
```

### Run Backend

```bash
python app.py
```

### Start Ollama

Make sure Ollama is installed and Llama 3 is available.

```bash
ollama run llama3
```

---

## Example Workflow

1. Upload scanned or digital document
2. Document is processed and OCR text is extracted
3. Metadata and semantic chunks are generated
4. User asks a natural language question
5. Relevant chunks are retrieved from ChromaDB
6. Llama 3 generates grounded response
7. Related document regions are highlighted
8. Final answer is returned with evidence

---

## Applications

* Academic document search
* Research document understanding
* Legal and administrative document analysis
* Institutional record systems
* Intelligent document assistants
* Digital libraries

---

## Future Improvements

* Multilingual document support
* Better reranking for retrieval
* Improved chunking strategies
* Table and figure understanding
* Faster vector retrieval
* Multimodal document reasoning

---


## Contributors

**Project Akshar Team**

- Manas Mondal  
- Arkaprava Jas  
- Anindya Midhey  
- Srijan Sarkar


---

## License

This project is intended for educational and research purposes.

```
