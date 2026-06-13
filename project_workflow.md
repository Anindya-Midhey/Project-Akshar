# Project Akshar Complete Workflow (Actual State)

Here is the flowchart representing the *actual* implemented workflow of the Project Akshar AI document processing system based on the current codebase. 

The pipeline orchestrator automatically routes documents either to the Image Workbench (Module 2) or directly to OCR, depending on whether they are scanned or digital.

```mermaid
graph TD
    %% Define Styles
    classDef module2 fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px;
    classDef module3 fill:#e8f5e9,stroke:#4caf50,stroke-width:2px;
    classDef module4 fill:#fff3e0,stroke:#ff9800,stroke-width:2px;
    classDef user fill:#eceff1,stroke:#607d8b,stroke-width:2px;
    
    %% Input
    User((User Input)):::user --> Upload[POST /pipeline/upload]
    Upload --> Detect[POST /pipeline/detect - Detect Document Type]
    
    %% Branching Logic (Actual)
    Detect -->|Scanned PDF / Image| M2[Module 2: Image Workbench]:::module2
    Detect -->|Digital PDF| DirectOCR[Direct OCR Route]
    
    %% Module 2 Workflow
    subgraph "Module 2: Image Workbench"
        M2 --> Transform[Transform & Crop]
        Transform --> Corners[Corner Detection]
        Corners --> Deskew[Dewarp & Deskew]
        Deskew --> Enhance[Enhance: Otsu / Adaptive Filtering]
    end
    
    %% Pipeline Convergence
    Enhance --> Converge((Processed Pages))
    DirectOCR --> Converge
    
    Converge --> ConvertPDF[POST /pipeline/convert-to-pdf]
    
    %% Module 3 Workflow
    subgraph "Module 3: OCR Layout Extraction"
        ConvertPDF --> RunOCR[POST /pipeline/run-ocr]
        RunOCR --> BBox[Extract Bounding Boxes & Text]
        BBox --> Layout[Layout Parsing & Visual Grounding]
    end
    
    %% Module 4 Workflow
    subgraph "Module 4: ChromaDB RAG QA System"
        Layout --> Index[POST /pipeline/index]
        Index --> Embed[Generate Embeddings]
        Embed --> ChromaDB[(ChromaDB Vector Store)]
        
        UserQuery((User Query)):::user --> Query[POST /pipeline/query]
        Query --> Retrieve[Retrieve Chunks from ChromaDB]
        ChromaDB -.-> Retrieve
        Retrieve --> Answer[Generate Answer with Bounding Box References]
    end
    
    %% Output
    Answer --> FinalOutput((Final Response & Visual Highlights)):::user
```

### Module Breakdown
- **Module 2 (Image Workbench)**: Handles all scanned documents and images. Offers interactive transformations, smart corner detection, dewarping, deskewing, and image enhancement.
- **Module 3 (OCR & Layout)**: Performs OCR to extract text and bounding boxes for visual grounding. Digital PDFs go straight here.
- **Module 4 (RAG QA)**: Embeds the OCR output into a ChromaDB vector store for answering user queries with bounding-box citations.
