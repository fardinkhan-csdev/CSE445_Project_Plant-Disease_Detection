# Domain Shift in Plant Disease Recognition

[1] **Kun Xiang *et al.*, “Quantifying the reliability gap in cross-domain plant disease classification…”, *Frontiers in Plant Science*, 2026.  
This study fine-tunes ResNet-50 on clean PlantVillage images and evaluates on field PlantDoc images. It **measures the domain gap**: a ResNet-50 achieves *99.73%* accuracy on PlantVillage but only *32.05%* on PlantDoc – a 67.7-point drop. Standard unsupervised adaptation (e.g. AdaBN, moment matching) modestly improves PlantDoc accuracy (to 0.343 or 0.366) using only unlabeled target data. No target labels are used for the baseline transfer. 

- **KEY RESULT:** Source (PlantVillage) accuracy 99.73% → Target (PlantDoc) accuracy 32.05%.  
- **Dataset:** PlantVillage (lab) → PlantDoc (field).  
- **Target labels used:** No (only unsupervised adaptation).  
- **Relevance:** Provides a concrete benchmark for the clean→noisy gap in plant disease classification. Directly relevant to PlantVillage→PlantDoc transfer. Suggests vanilla training (without adaptation) fails catastrophically. Applicable to our EfficientNet-B0 setup: we can similarly fine-tune EfficientNet on clean images and expect a large drop on real images. Techniques like **AdaBN** (adaptive BatchNorm, which re-estimates BN stats on target) could be implemented via LoRA layers to adjust normalization parameters without full finetuning.  
- **Quotable:** “A fine-tuned ResNet-50 suffered a 67.7 percentage-point accuracy collapse (99.73→32.05%) when moving from lab to field (PlantVillage→PlantDoc).”  
- **Priority:** [ESSENTIAL]

[2] **Wu *et al.*, “From Laboratory to Field: Unsupervised Domain Adaptation for Plant Disease Recognition in the Wild,” *Plant Phenomics*, 2023.  
Proposes **MSUN**, a multi-representation subdomain adaptation network with uncertainty regularization. They train only on labeled lab images (PlantVillage-like) and exploit *unlabeled* field images via UDA. Without source supervision, MSUN aligns feature distributions. As a result it achieves **56.06%** accuracy on unlabeled PlantDoc, compared to much lower baselines (e.g. ~30% without adaptation). 

- **KEY RESULT:** With MSUN, PlantDoc (target) accuracy is 56.06% (PlantDoc) versus very low <30% for a non-adapted model.  
- **Dataset:** Likely PlantVillage as source, PlantDoc as target (cross-species adaptation).  
- **Target labels used:** No (unsupervised DA with unlabeled target images).  
- **Relevance:** Demonstrates an unsupervised adaptation method tailored to plant disease (no target labels). It uses only source labels and target images. This matches our requirement for improving test performance on unlabeled target images. For EfficientNet-B0, one could adapt a frozen pretrained backbone with MSUN modules. In a PEFT context, one could imagine adding low-rank LoRA modules to implement the MSUN feature transformations (e.g. subdomain clusters) while fine-tuning only those modules.  
- **Quotable:** “MSUN was experimentally validated to achieve optimal results on PlantDoc…with accuracy 56.06%.”  
- **Priority:** [ESSENTIAL]  

[3] **Jeon *et al.*, “Bridging the Lab-to-Field Gap in Plant Disease Diagnosis through UDA Enhanced by Background Recomposition,” *SSRN preprint*, 2025.** (Not peer-reviewed)  
They introduce a two-step method for lab→field plant disease: first **Field-adaptive Background Recomposition (FBR)** – synthetically paste real-field backgrounds onto lab images – then apply unsupervised domain adaptation. Experiments on multiple crops (tomato, grape, etc.) show **robust field accuracy without any labeled target data**. 

- **KEY RESULT:** (Abstract only) Achieves robust field accuracy on tomato, chili, grape, apple tasks with no target labels.  
- **Datasets:** Various leaf disease datasets (tomato, grape, etc.) for indoor (source) vs. field (target).  
- **Target labels used:** No (fully unsupervised adaptation).  
- **Relevance:** Directly addresses lab-to-field transfer in plant disease. Although exact numbers aren’t given, the approach (augment lab images to mimic field) is pertinent. Background recomposition is a kind of data augmentation – this could be preprocessed before training EfficientNet. PEFT methods: applying LoRA is not directly needed here, since FBR is data-level. However, one could freeze EfficientNet and train small LoRA modules to refine features after applying FBR-augmented data.  
- **Quotable:** “Our approach first applies field-adaptive background recomposition for image augmentation, followed by unsupervised domain adaptation, enabling effective disease diagnosis in real environments.”  
- **Priority:** [STRONG] (innovative domain augmentation, but no exact accuracies)

[4] **Quilondrino *et al.*, “Mitigating Accuracy Loss in Plant Disease Detection: A Comparative Study of Multi-Stage Hybrid Classification Frameworks,” 2025 (manuscript).  
This work fuses CNNs with handcrafted texture features (GLCM). A key result: under their *Augmented* framework, **ConvNeXt** achieves *99.57%* on PlantVillage and *91.47%* on PlantDoc. In contrast, a vanilla ResNet-50 “drops from over 99% on PlantVillage to below 30% on PlantDoc” without adaptation. 

- **KEY RESULT:** (Augmented model) 99.57% PV → 91.47% PD. (Baseline ResNet: >99%→<30%.)  
- **Dataset:** PlantVillage (lab) and PlantDoc (field), focusing on tomato and corn diseases.  
- **Target labels used:** No (they propose architectures, not using target labels for training).  
- **Relevance:** Confirms enormous gap (99→30) and shows a hybrid method can nearly close it. Supports using multiple cues to generalize. The high PlantDoc accuracy implies that combining “global” CNN features with “local” GLCM via an ensemble can mitigate overfitting to clean patterns. For EfficientNet-B0, one could attach LoRA adapters to the final layers to integrate GLCM outputs or to mimic a similar ensemble effect (e.g. train a small LoRA layer that takes GLCM+feature concat to produce final logits).  
- **Quotable:** “ConvNeXt under the Augmented framework achieved 99.57% on PlantVillage and 91.47% on PlantDoc, significantly reducing the accuracy gap.”  
- **Priority:** [ESSENTIAL] (explicit PV→PD numbers, hybrid method illustration)

[5] **Li *et al.*, “Revisiting Batch Normalization for Practical Domain Adaptation,” *AAAI*, 2016.  
Introduces **Adaptive Batch Normalization (AdaBN)**: at test time, recompute the BatchNorm statistics on target-domain data without updating weights. This simple change “achieves deep adaptation effect… with no additional components, and is parameter-free”. 

- **KEY RESULT:** (No direct numeric in snippet) But AdaBN alone often closes a large part of the gap in DA tasks.  
- **Datasets:** (ImageNet, Office etc – general CV benchmarks)  
- **Target labels used:** No (uses only unlabeled target to recompute BN stats).  
- **Relevance:** Very practical adaptation: given an EfficientNet-B0 trained on lab images, one can run a few forward passes on field images to gather new batch-norm means/variances, and then test. This requires no label or training. In a PEFT context, AdaBN could be emulated by learning to shift the existing BN weights via a small LoRA module (e.g. learning an affine adjustment). In particular, QLoRA/LoRA could fine-tune just the BN *scale/shift* parameters cheaply.  
- **Quotable:** “Adaptive Batch Normalization… modulating the statistics in all BatchNorm layers… achieves deep adaptation… with no additional parameters.”  
- **Priority:** [STRONG]

[6] **Zhou *et al.*, “Domain Generalization with MixStyle,” *ICLR*, 2021.  
Presents **MixStyle**, a simple domain generalization trick: randomly mix feature map statistics (mean/variance) between instances during training to simulate new domains. This expands domain diversity (e.g. “photo vs sketch”). MixStyle is inserted in early CNN layers. It requires *only source data* at train time. They show it improves robustness to unseen domains across tasks. 

- **KEY RESULT:** (No cross-domain numbers here) MixStyle significantly boosts accuracy in cross-domain classification benchmarks (e.g. PACS, Office-Home).  
- **Datasets:** Standard DG benchmarks (e.g. PACS, VLCS).  
- **Target labels used:** No (source-only technique).  
- **Relevance:** For our problem, MixStyle could be applied during EfficientNet training on clean images to simulate variability (e.g. mixing styles between different leaves). This may narrow the lab-vs-field gap. It’s a training-time augmentation, so LoRA/QLoRA doesn’t directly factor in. However, one could pretrain EfficientNet with MixStyle on PV, then attach LoRA adapters to further adapt to field features. All three PEFT methods (LoRA/QLoRA/QKLoRA) could use MixStyle-augmented training in their fine-tuning phase to generalize better.  
- **Quotable:** “MixStyle… mixes instance-level feature statistics of training samples across source domains… synthesizing novel domains… increasing domain diversity.”  
- **Priority:** [SUPPORTING]

[7] **Tobin *et al.*, “Domain Randomization for Transferring Deep Networks from Simulation to the Real World,” *IROS*, 2017.  
In robotics, domain randomization trains entirely on synthetic images with randomized textures/colors. Tobin *et al.* show that with enough randomization, a model trained *only on non-realistic simulated images* can accurately locate objects in the real world (to within 1.5 cm). 

- **KEY RESULT:** Model trained on randomized simulated images transferred to real robot, accurate to 1.5 cm, with no real images used.  
- **Dataset:** Simulated object images (with random textures) → real-world robot scenes.  
- **Target labels used:** No (only simulated labels).  
- **Relevance:** Demonstrates that extensive source augmentation (domain randomization) can drastically reduce sim→real gap. Analogously, we could heavily randomize PlantVillage (e.g. random lighting, occlusions) during EfficientNet training. In a PEFT view, one could freeze the backbone and attach LoRA modules that are trained on randomizations of PV to mimic this effect. This suggests that broad style variation in training data can help a model see field-like images as “just another random variant.”  
- **Quotable:** “A model can be trained on simulated images with *non-realistic random textures* and still work in the real world (accurate to 1.5 cm).”  
- **Priority:** [STRONG] (general sim→real analog)

[8] **Hermann *et al.*, “The Origins and Prevalence of Texture Bias in CNNs,” *NeurIPS*, 2020.  
Analyzes why CNNs are texture-biased. The authors note that ImageNet-trained CNNs “make classifications based on superficial textural features rather than the shape information… used by humans”. They find data augmentation can increase shape bias, but most standard models remain texture-biased. 

- **KEY RESULT:** (Summary) Standard CNNs “prefe­r texture over shape, making it difficult to generalize to different distributions”.  
- **Dataset:** ImageNet variants (Stylized-ImageNet, shape-texture cue conflict images).  
- **Target labels used:** N/A (analysis paper).  
- **Relevance:** The texture-vs-shape bias explains part of why a model trained on clean lab images (with uniform backgrounds and high-quality textures) fails on noisy field images (different textures, lighting). It suggests that encouraging shape-based features (through augmentation or style mixing) could improve OOD robustness. In terms of PEFT, one could fine-tune an EfficientNet’s early layers with LoRA to reduce texture reliance (e.g. using stylized images).  
- **Quotable:** “CNNs appear to make classifications based on superficial textural features… rather than the shape information… making it difficult for models to generalize to different distributions.”  
- **Priority:** [ESSENTIAL] (theoretical insight)

[9] **Liang *et al.*, “Do We Really Need to Access Source Data? Source Hypothesis Transfer for UDA (SHOT),” *ICML*, 2020.  
Proposes **SHOT**, a source-free DA method. SHOT freezes the source classifier (“hypothesis”) and adapts the feature extractor on unlabeled target data via entropy maximization and self-supervised pseudo-labeling. It achieves SOTA performance on various DA benchmarks *without using source data* at adaptation time. 

- **KEY RESULT:** (No specific numbers here) SHOT matches or exceeds prior DA methods on multiple benchmarks by aligning target features to the fixed source classifier.  
- **Dataset:** Standard DA benchmarks (e.g. Office-31, VisDA).  
- **Target labels used:** No (unlabeled target only).  
- **Relevance:** Illustrates a modern approach to target-unlabeled adaptation for classification. For our case, one could freeze the EfficientNet classifier head and use target images with entropy minimization/self-training to adjust the backbone via PEFT (e.g. LoRA layers). All three LoRA variants can be used to update only a small set of parameters during this adaptation, keeping most of the EfficientNet fixed.  
- **Quotable:** “SHOT freezes the classifier and learns the target-specific feature extractor by exploiting information maximization and self-supervised pseudo-labeling to align target representations to the source hypothesis.”  
- **Priority:** [SUPPORTING]

[10] **Geirhos *et al.*, “ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness,” *ICLR*, 2019.  
Through a “cue conflict” test (e.g. elephant-textured knife), they show ImageNet CNNs almost always predict the texture label. “CNNs are strongly biased towards recognising textures rather than shapes”. They further show that training on a *stylized* ImageNet (texture randomized) makes CNNs more shape-biased and robust to distortions. 

- **KEY RESULT:** (Psychophysics) ResNet-50 is texture-biased on normal ImageNet, but can learn shape-bias if trained on stylized images.  
- **Dataset:** ImageNet and Stylized-ImageNet (for shape vs texture).  
- **Target labels used:** N/A.  
- **Relevance:** Reinforces that standard CNN features latch onto texture cues. Real-world field images often break these texture patterns (different backgrounds, lower resolution), so a texture-biased model performs poorly. Suggests remedy: train with stylized or texture-mixed images. In our work, one could pretrain EfficientNet-B0 with style augmentations. PEFT could then fine-tune on field-like styles by adding LoRA layers that encourage shape features.  
- **Quotable:** “ImageNet-trained CNNs are strongly biased towards recognising textures rather than shapes”.  
- **Priority:** [ESSENTIAL] (fundamental cause)

Each of these papers provides evidence or methods relevant to our lab-to-field transfer problem. The domain-gap quantification papers ([1],[4]) highlight the severity of the issue. The adaptation methods ([2],[5],[9]) offer strategies (unsupervised alignment, test-time tuning) that require no target labels. Domain-generalization and augmentation works ([6],[7]) show how to diversify training to better mimic field conditions. Finally, theory papers ([8],[10]) explain *why* clean-trained CNNs fail (texture bias, covariate shift), guiding us toward remedies. **LoRA/QLoRA/QKLoRA** can incorporate many of these ideas by acting as lightweight fine-tuning layers: e.g., using LoRA modules to adapt BatchNorm (AdaBN), to implement MixStyle at train time, or to adjust features during SHOT-style adaptation, all while keeping the bulk of EfficientNet frozen. 

**Sources:** The above summaries and numbers are drawn from the cited literature.