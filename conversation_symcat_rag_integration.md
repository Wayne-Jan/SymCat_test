# SymCat RAG Integration Discussion

## Project Overview Context

The SymCat project is a plant disease categorization and classification system that uses a multi-modal approach combining both image and text data. It leverages visual-semantic information to create a robust classification system for plant diseases.

## Key Discussion Points

### Integrating RAG with SymCat

**Original Idea**: Combine a Retrieval-Augmented Generation (RAG) system with SymCat to enhance diagnostic capabilities.

**Core Insight**: Use predicted disease traits as RAG queries rather than just returning disease categories, creating a powerful hybrid system that addresses the fundamental challenge in plant disease diagnostics where certainty isn't always possible from a single image.

### Conceptual Workflow:
1. Multi-modal model identifies visual traits and symptom patterns from images
2. These traits become structured queries
3. RAG system retrieves relevant information about possible diseases matching those traits
4. Users receive both trait identification AND contextual information about potential diseases

### Evaluation Approaches

Multiple evaluation approaches were discussed:

1. **Trait Classification Performance (Direct Metrics)**
   - Show precision/recall/F1 for trait identification
   - Establishes technical foundation

2. **Disease Diagnosis Improvement (End Task)**
   - Compare LLM diagnosis accuracy using:
     - Photo only
     - Photo + identified traits
   - Demonstrates practical utility

3. **Information Quality Analysis**
   - Compare information completeness:
     - Conventional: "This is Leaf Rust (78% confidence)"
     - Trait-based: "Circular yellow spots (93%), along leaf veins (87%), on older leaves (78%) - consistent with early Leaf Rust"
   - Showcases enriched diagnostic experience

4. **Uncertainty Handling**
   - Create deliberately ambiguous test cases
   - Show how the system provides valuable information even when exact disease classification is uncertain

### Expert Knowledge Base Alternative

Instead of building a full RAG system, the discussion revealed that the existing `symptom_combination_diagnostic_value.json` file provides a structured expert-knowledge database that maps specific symptom combinations to diseases with diagnostic confidence values.

Benefits of this approach:
1. **Evidence-based diagnosis** - Each diagnosis has specificity and sensitivity metrics
2. **Transparent reasoning** - Context descriptions provide clear explanations
3. **Confidence scoring** - Diagnostic value field quantifies certainty
4. **Multilingual support** - Descriptions in multiple languages

### Final Experimental Design

A clean comparison methodology was agreed upon:

1. Send a photo to a multimodal LLM, asking it to identify the disease category
2. Use the SymCat system to extract disease traits, then send both the photo and the traits to the same LLM
3. Compare the diagnostic accuracy between the two approaches

This approach was deemed methodologically sound because it:
- Isolates just one variable (the presence of trait information)
- Provides a direct performance measure
- Reflects real-world applicability

## Publication Strategy

For publication in the "Generative AI in Smart Agriculture" special issue (deadline: December 31, 2025), the project shows strong potential by:

1. Aligning with the call for papers' focus on multi-modal approaches to agricultural problems
2. Offering novel contributions through trait extraction and diagnostic pathways
3. Addressing a key challenge in the field (handling uncertainty in plant disease diagnosis)

## Next Steps

1. Design a multiple-choice test with diverse plant disease images
2. Set up the comparative experiment between image-only and image+traits diagnosis
3. Analyze results focusing on both accuracy and information quality
4. Prepare manuscript highlighting how trait extraction improves both diagnosis accuracy and explanation quality