RESEARCH PROPOSAL

**MY EYE, MY BRAIN**

*Development and Validation of an Accessible AI-Enabled Retinal Screening System for the Early Detection of Ocular and Systemic Disease in Underserved South African Populations*

by

**Samuel Sephiri**

Qualifications: MBA · BSc Honours · BSc Mathematical Sciences

Submitted in partial fulfilment of the requirements for the degree

**DOCTOR OF PHILOSOPHY**

in the field of \[Computer Science / Engineering / Health Sciences --- to be confirmed with prospective supervisor\]

FACULTY OF \[TO BE CONFIRMED\]

**UNIVERSITY OF PRETORIA / UNIVERSITY OF THE WITWATERSRAND**

Prospective Supervisor:

Date of submission: TBC

# Table of Contents

[Table of Contents [1](#table-of-contents)](#table-of-contents)

[1. Abstract [1](#abstract)](#abstract)

[2. Introduction and Background [1](#introduction-and-background)](#introduction-and-background)

[3. Problem Statement and Research Gap [1](#problem-statement-and-research-gap)](#problem-statement-and-research-gap)

[4. Research Aim and Objectives [1](#research-aim-and-objectives)](#research-aim-and-objectives)

[4.1 Aim [1](#aim)](#aim)

[4.2 Objectives [1](#objectives)](#objectives)

[5. Research Questions [1](#research-questions)](#research-questions)

[5.1 Primary research question [1](#primary-research-question)](#primary-research-question)

[5.2 Secondary research questions [1](#secondary-research-questions)](#secondary-research-questions)

[6. Literature Review [1](#literature-review)](#literature-review)

[6.1 AI-based retinal disease detection [1](#ai-based-retinal-disease-detection)](#ai-based-retinal-disease-detection)

[6.2 AI screening in African and low-resource contexts [1](#ai-screening-in-african-and-low-resource-contexts)](#ai-screening-in-african-and-low-resource-contexts)

[6.3 Low-cost portable fundus imaging [1](#low-cost-portable-fundus-imaging)](#low-cost-portable-fundus-imaging)

[6.4 The retina as a marker of systemic and brain health [1](#the-retina-as-a-marker-of-systemic-and-brain-health)](#the-retina-as-a-marker-of-systemic-and-brain-health)

[6.5 Regulatory context [1](#regulatory-context)](#regulatory-context)

[7. Significance and Contribution of the Study [1](#significance-and-contribution-of-the-study)](#significance-and-contribution-of-the-study)

[8. Research Design and Methodology [1](#research-design-and-methodology)](#research-design-and-methodology)

[8.1 Phase 1 --- Data assembly and model development [1](#phase-1-data-assembly-and-model-development)](#phase-1-data-assembly-and-model-development)

[8.2 Phase 2 --- Integrated field validation [1](#phase-2-integrated-field-validation)](#phase-2-integrated-field-validation)

[8.3 Phase 3 --- Analysis, thesis writing, and dissemination [1](#phase-3-analysis-thesis-writing-and-dissemination)](#phase-3-analysis-thesis-writing-and-dissemination)

[9. Population, Sampling, and Data Sources [1](#population-sampling-and-data-sources)](#population-sampling-and-data-sources)

[10. Ethical Considerations [1](#ethical-considerations)](#ethical-considerations)

[11. Research Timeline [1](#research-timeline)](#research-timeline)

[12. Translation and Impact Pathway [1](#translation-and-impact-pathway)](#translation-and-impact-pathway)

[13. Scope and Limitations [1](#scope-and-limitations)](#scope-and-limitations)

[14. Preliminary Resource Requirements [1](#preliminary-resource-requirements)](#preliminary-resource-requirements)

[References (preliminary) [1](#references-preliminary)](#references-preliminary)

# 1. Abstract

This study proposes the design, development, and validation of an integrated, low-cost artificial intelligence (AI) enabled retinal screening system --- working title "My Eye, My Brain" --- intended to enable early detection of vision-threatening and systemically linked eye disease in underserved South African communities. The research addresses two closely coupled problems: first, the near-total absence of AI diagnostic models trained and validated specifically on African patient populations, despite growing evidence that models trained on non-African data may not generalise reliably to local populations, imaging hardware, and clinical workflows; and second, the lack of an affordable, portable retinal imaging device suited to deployment in rural and peri-urban primary care and pharmacy-based clinics.

The study is structured as two interdependent research streams. The first involves the curation, fine-tuning, and clinical validation of a deep learning model --- referred to throughout as "Sam" --- for the detection of diabetic retinopathy, glaucoma, and related retinal pathology, using a combination of established public benchmark datasets and newly collected South African patient data. The second involves the design and prototyping of a smartphone-based, low-cost fundus imaging device suitable for operation by non-specialist clinical staff. The two streams are validated jointly through an integrated field study that tests the model\'s performance on images captured by the study\'s own prototype device under real rural clinical conditions, directly addressing a well-documented failure mode in which AI systems validated on curated datasets underperform once deployed with different imaging workflows and hardware.

The study further maps its research outputs against South Africa\'s emerging regulatory framework for AI/ML-enabled medical devices, positioning the resulting evidence base to support future regulatory authorisation and translation into primary healthcare and retail pharmacy screening channels. Expected contributions include a validated, population-appropriate diagnostic model, an openly documented low-cost device design, and a body of clinical evidence that directly addresses a recognised gap in African-validated ophthalmic AI systems.

# 2. Introduction and Background

Retinal disease --- principally diabetic retinopathy (DR), glaucoma, and age-related macular degeneration --- remains one of the leading causes of preventable blindness worldwide. Deep learning-based screening tools have demonstrated diagnostic accuracy approaching that of expert human graders on curated benchmark datasets, and multiple systems (including IDx-DR, EyeArt, Medios AI, Google\'s ARDA, and Singapore\'s SELENA+) have progressed from research prototypes to deployed clinical tools. However, the vast majority of the data underlying these systems originates from the United States, India, Brazil, Singapore, and the Netherlands.

Sub-Saharan Africa carries a disproportionate burden of preventable vision loss relative to its access to specialist eye care, with the region having among the lowest ratios of ophthalmologists to population in the world. At the same time, a growing body of "oculomics" research indicates that retinal imaging offers a non-invasive window into systemic and neurological health, including cerebrovascular and neurodegenerative disease, because the retinal and cerebral microvasculature share anatomical and physiological similarities. This dual relevance --- to ocular health directly, and to systemic and brain health indirectly --- motivates the working title of this study.

This proposal is informed by a period of directed scoping work, including a review of existing public retinal imaging datasets, low-cost fundus camera designs, AI deployment case studies in African screening programmes (Tanzania, Zambia), and South Africa\'s newly published regulatory guidance for AI/ML-enabled medical devices. That scoping work forms the basis of the research gap, aim, and methodology set out below.

# 3. Problem Statement and Research Gap

Three converging gaps motivate this research:

1.  Population representation gap: no AI system for diabetic retinopathy or glaucoma screening currently holds regulatory authorisation in South Africa on the basis of training or validation data drawn from an African population. Regulators and implementers alike have raised concerns that under-representation of African patients in training data may cause silent performance degradation when such systems are deployed locally.

2.  Deployment-context gap: existing evidence from real-world rollouts shows that AI screening systems validated on curated, high-quality datasets can underperform once deployed in field conditions where imaging workflows, camera hardware, lighting, and operator skill differ from the original training environment. This is a device-and-model problem, not a model problem alone --- yet the two are rarely developed and validated together.

3.  Access gap: conventional fundus cameras are expensive, bulky, and require a trained photographer, making them impractical for rural or peri-urban primary care settings. Low-cost smartphone-based fundus camera designs exist in the literature but have not, to the candidate\'s knowledge, been paired with a model specifically fine-tuned and validated on South African patients.

Taken together, these gaps define a specific and under-served research problem: the joint design and validation of an affordable imaging device and a population-appropriate diagnostic model, tested together under real rural clinical conditions in South Africa.

# 4. Research Aim and Objectives

## 4.1 Aim

To develop and validate an accessible, AI-enabled retinal screening system --- comprising a population-appropriate diagnostic model and a low-cost imaging device --- suitable for the early detection of retinal and related systemic disease in underserved South African settings.

## 4.2 Objectives

4.  To assemble a representative training and validation dataset combining established public benchmark retinal image datasets with newly collected South African patient data.

5.  To develop and fine-tune a deep learning model ("Sam") for multi-class detection of diabetic retinopathy, glaucoma, and related retinal pathology, and to benchmark its performance against baseline models trained predominantly on non-African populations.

6.  To design and prototype a low-cost, smartphone-based fundus imaging device suitable for operation by non-specialist clinical staff in rural and peri-urban settings.

7.  To conduct an integrated field validation study evaluating diagnostic model performance specifically on images captured by the prototype device under real clinical conditions, including subgroup performance analysis.

8.  To evaluate the technical, clinical, and regulatory translation pathway required for future authorisation and deployment of the system within South African primary healthcare and retail pharmacy screening channels.

# 5. Research Questions

## 5.1 Primary research question

Can a lightweight deep learning model, fine-tuned on South African-representative retinal imaging data and paired with a low-cost, smartphone-based fundus camera, achieve clinically acceptable diagnostic accuracy for the early detection of diabetic retinopathy and glaucoma in rural South African populations?

## 5.2 Secondary research questions

-   How does the diagnostic performance of models trained primarily on non-African datasets change when fine-tuned and tested on South African patient data?

-   What device design parameters (optics, field of view, image resolution, illumination) are necessary to produce diagnostically usable fundus images at low cost, operable by non-specialist staff?

-   What is the effect of real-world field conditions (device hardware, operator variability, lighting) on model performance, relative to performance on curated benchmark datasets?

-   What are the principal technical, clinical, and regulatory barriers to deploying such a system within South Africa\'s primary healthcare and retail pharmacy screening infrastructure?

# 6. Literature Review

## 6.1 AI-based retinal disease detection

A substantial and mature literature now exists on deep learning approaches to retinal disease detection, encompassing convolutional neural networks and, more recently, vision transformer architectures. This includes single-disease detection (predominantly diabetic retinopathy and glaucoma), as well as multi-disease and multi-label frameworks. Reviews of this literature (e.g. Muchuchuti and Viriri, 2023, published from the University of KwaZulu-Natal) note that despite strong reported accuracy, model explainability and generalisability across populations remain open challenges.

## 6.2 AI screening in African and low-resource contexts

A smaller body of work has evaluated AI screening specifically within African contexts. A 2019 clinical validation study published in The Lancet Digital Health evaluated a deep learning ensemble --- trained on Singaporean data --- against a Zambian diabetic retinopathy screening population, reporting clinically acceptable sensitivity and specificity despite the training/testing population mismatch. Separately, an implementation study of an AI screening system within Tanzania\'s national diabetic eye screening programme documented the practical and regulatory considerations of introducing AI into a low-resource screening pathway, including a lack of African-trained systems and the operational value of offline functionality. A recent methodological paper on data-efficient AI for low-resource settings further documents a well-known case in which a diabetic retinopathy screening system underperformed once deployed in rural clinics, attributable to differences between the training environment and real-world imaging workflows --- a finding of direct relevance to the design of this study.

## 6.3 Low-cost portable fundus imaging

Multiple published designs demonstrate that diagnostically usable fundus photographs can be captured using smartphone-based cameras at a fraction of the cost of conventional office-based fundus photography systems. Chalam, Chamchikh and Gasparian (2022) describe a smartphone-mounted condensing-lens system producing high-resolution fundus images without requiring a trained photographer. Other published work addresses the narrow field-of-view limitation common to single-shot smartphone fundus cameras through miniaturised indirect ophthalmoscopy designs, extending usable field of view for peripheral disease detection.

## 6.4 The retina as a marker of systemic and brain health

An emerging body of "oculomics" research uses AI-based analysis of retinal images to assess systemic and neurological health, on the premise that retinal and cerebral microvasculature share structural similarities. Recent work has applied self-supervised graph embeddings to retinal images for early detection of cerebral small vessel disease, and a 2026 analysis in The Lancet Digital Health reviewed the broader potential of AI-based retinal imaging for brain health assessment, including dementia and stroke risk. This literature motivates the longer-term systemic health relevance of the proposed system, while the core thesis scope remains focused on ocular disease detection.

## 6.5 Regulatory context

South Africa\'s Health Products Regulatory Authority (SAHPRA) published its first consolidated guidance for AI/ML-enabled medical devices in September 2025. This guidance requires that manufacturers explicitly assess whether a device was trained on data from outside South Africa and whether it performs equally well on local populations --- a requirement that aligns directly with the research question of this study --- and specifies that devices materially influencing clinical decisions will typically be classified as moderate-to-high risk (Class C or D), requiring clinical evidence with subgroup performance analysis. This regulatory context directly shapes the study\'s methodology and validation design.

# 7. Significance and Contribution of the Study

This study makes contributions at three levels. Academically, it addresses a specific and documented gap in the literature: the near-absence of retinal disease detection models developed and validated jointly with their intended low-cost imaging hardware, on African patient data. Clinically, it produces an evidence base directly relevant to improving early detection of preventable blindness in a population currently underserved by specialist ophthalmic care. Translationally, the study is deliberately designed so that its clinical validation evidence can inform a future regulatory submission and deployment pathway, extending the value of the research beyond the thesis itself without displacing the thesis\'s core scientific contribution.

# 8. Research Design and Methodology

The study adopts an applied, mixed-methods design combining quantitative model performance evaluation, engineering design and prototyping, and field-based validation with a qualitative usability assessment component. The methodology is organised into three phases.

## 8.1 Phase 1 --- Data assembly and model development

A composite training dataset will be assembled from established public benchmark datasets, including EyePACS, APTOS, IDRiD, Messidor, AIROGS, RFMiD, and BRSET, selected to cover diabetic retinopathy, glaucoma, and multi-disease classification with varying annotation depth (image-level labels through to pixel-level lesion annotation). This will be supplemented with newly collected South African patient data, gathered in partnership with a participating clinical site, subject to institutional ethics approval. The candidate\'s model ("Sam") will be developed and fine-tuned on this composite dataset and benchmarked against comparable architectures trained only on non-African data, to directly quantify any performance shift attributable to population-specific fine-tuning.

In parallel, a low-cost, smartphone-based fundus imaging device will be designed and prototyped, drawing on published low-cost optical designs, with attention to field of view, image resolution, and ease of operation by non-specialist clinical staff.

## 8.2 Phase 2 --- Integrated field validation

The model and prototype device will be validated jointly in a field study at one or more participating clinical sites. Images captured with the prototype device will be evaluated against expert human grading (reference standard) and, where feasible, against images from conventional fundus photography. Performance will be assessed using standard diagnostic metrics (sensitivity, specificity, AUC), with subgroup analysis by relevant demographic and imaging-condition variables, in line with South African regulatory guidance for AI/ML-enabled medical devices. A structured usability assessment with non-specialist device operators will also be conducted.

## 8.3 Phase 3 --- Analysis, thesis writing, and dissemination

Findings will be statistically analysed and written up as the doctoral thesis, with results disseminated through peer-reviewed publication. Documentation produced during Phases 1 and 2 (technical specifications, validation data, subgroup analyses) will be structured so that it can subsequently inform a regulatory submission, without this submission itself forming part of the doctoral scope.

# 9. Population, Sampling, and Data Sources

The target population for field validation comprises adult patients attending participating rural and peri-urban primary care or ophthalmology clinics in South Africa. A purposive/convenience sampling approach is proposed for the field validation phase, with sample size to be determined by a formal power calculation once baseline model performance from Phase 1 is established. Data sources comprise (a) public benchmark datasets, used under their respective data-use terms, and (b) newly collected, ethically approved South African patient data, de-identified prior to model development.

# 10. Ethical Considerations

-   Institutional ethics approval will be sought from the relevant university Research Ethics Committee (Engineering/Health Sciences, as applicable) prior to any data collection.

-   All patient data collection will comply with the Protection of Personal Information Act (POPIA), with images and associated clinical data de-identified prior to storage and model development.

-   Informed consent will be obtained from all participants in the field validation study, including clear disclosure that the device is an investigational research tool and not (at this stage) a clinically authorised diagnostic system.

-   A data management plan will specify secure storage, access control, and retention periods for all patient-derived data, consistent with university and national requirements.

-   Findings indicating a positive disease screening result will be referred through an appropriate clinical pathway at the participating site; the study will not rely on AI output alone for any clinical decision.

# 11. Research Timeline

The table below sets out an indicative timeline for the doctoral study, structured around the two research streams described in Section 8. Regulatory and commercial translation activities (shown for context in the final column) fall outside the formal thesis scope but are informed directly by the thesis outputs.

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Period**                      **PhD research activities**                                                                                                       **Related translation milestone (context only)**
  ------------------------------- --------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------
  Year 1 (H1 2027 -- H2 2027)     Literature review; ethics application; data assembly; baseline model development; device prototype design                         Establishment licence and quality management system groundwork (parallel, non-thesis activity)

  Year 2 (H1 2028 -- H2 2028)     Model fine-tuning and benchmarking; device prototype build and bench testing; field validation study design and ethics approval   Regulatory dossier preparation and retailer/clinic partnership discussions

  Year 3 (H1 2029 -- H2 2029)     Integrated field validation study; subgroup performance analysis; usability assessment; data analysis                             Regulatory submission and review process

  Year 3.5--4 (H2 2029 -- 2030)   Thesis writing, internal review, examination, and peer-reviewed publication                                                       Pilot deployment planning informed by thesis findings
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 12. Translation and Impact Pathway

Beyond its academic contribution, this study is designed so that its validated model and device documentation can directly inform a future regulatory submission to South Africa\'s Health Products Regulatory Authority, and, subject to that authorisation, a phased deployment pathway through primary healthcare facilities and in-store pharmacy wellness clinics. This translation pathway is presented here for context and motivation only; it is not part of the assessed doctoral research and will be pursued as a separate, post-thesis commercialisation initiative, most likely through a university technology transfer office or spin-out structure.

# 13. Scope and Limitations

-   The doctoral study will produce a validated research prototype and clinical evidence base; it will not itself deliver a regulatory-approved, market-ready medical device.

-   Field validation will be limited to a defined number of participating clinical sites and will not constitute a nationally representative epidemiological sample.

-   The study focuses primarily on diabetic retinopathy and glaucoma detection; broader systemic/brain health applications of the retinal data (oculomics) are noted as a motivating context and potential area of future work, not a core deliverable of this thesis.

# 14. Preliminary Resource Requirements

Indicative resource requirements are set out below and will be refined in consultation with the supervisor and faculty research office, including exploration of bursary and grant funding (e.g. NRF, SAMRC, university postgraduate bursaries).

-   Device prototyping components and iterations (optics, mounting hardware, smartphone/compute unit)

-   Cloud or local compute resources for model training and benchmarking

-   Ethics application and data collection costs at participating clinical site(s)

-   Conference travel and peer-reviewed publication costs

-   Statistical/technical software licensing, where not provided by the university

# References (preliminary)

Chalam, K.V., Chamchikh, J. and Gasparian, S. (2022) \'Optics and utility of low-cost smartphone-based portable digital fundus camera system for screening of retinal diseases\', Diagnostics, 12(6), 1499.

Muchuchuti, S. and Viriri, S. (2023) \'Retinal disease detection using deep learning techniques: a comprehensive review\', Journal of Imaging, 9(4), 84.

Xia, X., Li, Y., Xiao, G., Zhan, K., Yan, J., Huang, Z., Huang, G. and Fang, Y. (2024) \'Benchmarking deep models on retinal fundus disease diagnosis and a large-scale dataset\', Signal Processing: Image Communication, 127, 117151.

Beede, E., Baylor, E., Hersch, F., Iurchenko, A., Wilcox, L., Ruamviboonsuk, P. and Vardoulakis, L.M. (2020) \'A human-centered evaluation of a deep learning system deployed in clinics for the detection of diabetic retinopathy\', Proceedings of the 2020 CHI Conference on Human Factors in Computing Systems.

Scientific Reports (2026) \'Deep learning for early detection of cerebral small vessel disease using self-supervised graph embeddings and retinal image analysis\' \[full author list and citation details to be verified\].

South African Health Products Regulatory Authority (2025) Communication to Industry: Regulatory requirements for AI/ML-enabled medical devices and IVDs (MD08-2025/2026). Pretoria: SAHPRA.

Bellemo, V. et al. (2019) \'Artificial intelligence using deep learning to screen for referable and vision-threatening diabetic retinopathy in Africa: a clinical validation study\', *The Lancet Digital Health*, 1(1), pp. e35--e44. doi: 10.1016/S2589-7500(19)30004-4.

Cleland, C.R., Makupa, W.U., Shilio, B.R., Rwiza, J., Macleod, D., Bascaran, C. and Burton, M.J. (2025) \'Implementing an artificial intelligence system into a diabetic eye screening programme in Tanzania\', *Transactions of The Royal Society of Tropical Medicine and Hygiene*, 119(6), pp. 567--569. doi: 10.1093/trstmh/trae132.

Katende, R. (2025) \'Rethinking data-efficient artificial intelligence for low-resource settings\', *Machine Learning with Applications*, 100796. doi: 10.1016/j.mlwa.2025.100796.
