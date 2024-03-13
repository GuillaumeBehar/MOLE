# IMPORTATION
import json
import random
import os


def get_main_dir(depth: int = 0):  # nopep8
    """Get the main directory of the project."""
    import os
    import sys
    from os.path import dirname as up
    main_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(depth):
        sys.path.append(up(main_dir))
        main_dir = up(main_dir)
    return main_dir


MAIN_DIR_PATH = get_main_dir(1)  # nopep8

from utils.lecture_xml import pmid_to_pmcid, get_data
from pmcqa_evaluate import get_instance_from_pubid

# Obtenez le chemin absolu du répertoire du script Python
repertoire_script = os.path.dirname(os.path.abspath(__file__))

# Construisez le chemin complet pour le fichier JSON
chemin_fichier_json = os.path.join(repertoire_script, "pubID_list.json")

with open(chemin_fichier_json, "r") as file:
    # Charger les données JSON depuis le fichier
    data = json.load(file)

# Construction de la liste des choix, où tout les articles sont dans PMC et sont récupérables

# liste_id = data['pubid_list']
# choix = []
# i = 0
# bon = 0

# while len(choix) < 50:
#     article = random.choice(liste_id)
#     print("PMID :", article)
#     liste_id.remove(article)
#     if ((id := pmid_to_pmcid(article)) != None):
#         bon += 1
#         print("PMCID :", id)
#         data = get_data(id, api=True, show=True)
#         if (data != None):
#             choix.append(article)
#     i += 1
#     if i == 10:
#         print(bon)
#         print(len(choix))
#         i = 0

# print(choix)

# Retrouver les id de PMC à partir de ceux de PM

id_test_pm = [22022005, 21639875, 20852029, 25844699, 25379003, 26817669, 22563393, 20659337, 27643685, 26693009, 19888227, 20878146, 26337974, 23355459, 25495800, 22640485, 24059973, 24409166, 22909062, 24447369, 19180231, 22569336, 23231769, 23557178,
              21617180, 24958351, 27500275, 19933996, 24330812, 26227965, 27574676, 27473420, 22709483, 26289293, 23949151, 27336604, 26460750, 18575589, 24884655, 18493326, 23015864, 26175775, 26418562, 26418133, 21696606, 25036418, 24847033, 26295946, 27595989, 21981946]

id_test_pmc = [3195132, 3218944, 2992784, 4467582, 4213846, 4729033, 3338521, 2918533, 5028067, 4676151, 2788266, 3020317, 4558969, 3721168, 4279816, 3511263, 3851161, 3873600, 3441385, 3903047, 2628736, 3446336, 3551651, 3651302,
               3121440, 4230643, 4975560, 2828649, 4029378, 4521469, 4964010, 4966758, 3416570, 4895690, 3776978, 4947703, 4603675, 2424173, 4041915, 2375056, 3445072, 4499589, 4587925, 4587927, 3144447, 4103805, 4309068, 4546589, 5011830, 3199253]

questions = ['Does assessment of antidiabetic potential of Cinnamomum tamala leave extract in streptozotocin induced diabetic rats?', 'Are elevated pre-treatment levels of plasma C-reactive protein associated with poor prognosis after breast cancer : a cohort study?',
             'Does candesartan attenuate diabetic retinal vascular pathology by restoring glyoxalase-I function?', 'Does xenon protect Against Septic Acute Kidney Injury via miR-21 Target Signaling Pathway?', 'Does rg3-enriched Korean Red Ginseng improve vascular function in spontaneously hypertensive rats?',
              'Does routine whole body CT of high energy trauma patients lead to excessive radiation exposure?', 'Is bronchial responsiveness related to increased exhaled NO ( FE ( NO ) ) in non-smokers and decreased FE ( NO ) in smokers?', 'Is inadequate glucose control in type 2 diabetes associated with impaired lung function and systemic inflammation : a cross-sectional study?',
               'Does assessment of Local Mosquito Species incriminate Aedes aegypti as the Potential Vector of Zika Virus in Australia?', 'Do supplemental oxygen users with pulmonary fibrosis perceive greater dyspnea than oxygen non-users?', 'Are circulating tumour cells associated with increased risk of venous thromboembolism in metastatic breast cancer patients?',
                "Does clinicians ' response to hyperoxia in ventilated patients in a Dutch ICU depend on the level of FiO2?", 'Does deep brain stimulation induce antiapoptotic and anti-inflammatory effects in epileptic rats?', 'Do network models of genome-wide association studies uncover the topological centrality of protein interactions in complex diseases?',
                 'Does spatial localization of the first and last enzymes effectively connect active metabolic pathways in bacteria?', 'Does oncolytic adenovirus armed with IL-24 inhibit the growth of breast cancer in vitro and in vivo?', 'Is the sodium channel gene family specifically expressed in hen uterus and associated with eggshell quality traits?',
                  'Does white matter disease correlate with lexical retrieval deficits in primary progressive aphasia?', 'Is induction of interleukin-10 dependent on p38 mitogen-activated protein kinase pathway in macrophages infected with porcine reproductive and respiratory syndrome virus?',
                   'Are gene expression and nucleotide composition associated with genic methylation level in Oryza sativa?', 'Is pECAM-independent thioglycollate peritonitis associated with a locus on murine chromosome 2?', 'Do crk adaptor proteins act as key signaling integrators for breast tumorigenesis?',
                    'Is alpha B-crystallin a new prognostic marker for laryngeal squamous cell carcinoma?', 'Is traumatic axonal injury in the mouse accompanied by a dynamic inflammatory response , astroglial reactivity and complex behavioral changes?', 'Does genetic defect in phospholipase Cδ1 protect mice from obesity by regulating thermogenesis and adipogenesis?',
                     'Does pP2A inhibition overcome acquired resistance to HER2 targeted therapy?', 'Is spontaneous Preterm Delivery , Particularly with Reduced Fetal Growth , Associated with DNA Hypomethylation of Tumor Related Genes?', 'Does examination of all type 2 diabetes GWAS loci reveal HHEX-IDE as a locus influencing pediatric BMI?',
                      'Is uric acid a risk factor for ischemic stroke and all-cause mortality in the general population : a gender specific analysis from The Tromsø Study?', 'Is protocadherin-1 a glucocorticoid-responsive critical regulator of airway epithelial barrier function?',
                       "Do retrospective review of a tertiary adult burn centre 's experience with modified Meek grafting?", 'Does formula PSORI-CM01 eliminate psoriasis by inhibiting the expression of keratinocyte cyclin B2?', 'Does an acidic oligopeptide displayed on AAV2 improve axial muscle tropism after systemic delivery?',
                        'Does antimicrobial Susceptibilities of Aerobic isolate from Respiratory Samples of Young New Zealand Horses?', 'Is loss of PTEN expression an independent predictor of favourable survival in endometrial carcinomas?', 'Does mendelian randomisation analysis strongly implicate adiposity with risk of developing colorectal cancer?',
                         'Does different MMSE Score be Associated with Postoperative Delirium in Young-Old and Old-Old Adults?', 'Does nitric oxide mediate stretch-induced Ca2+ release via activation of phosphatidylinositol 3-kinase-Akt pathway in smooth muscle?', 'Is cBD binding domain fused γ-lactamase from Sulfolobus solfataricus an efficient catalyst for ( - ) γ-lactam production?',
                          'Do global considerations in hierarchical clustering reveal meaningful patterns in data?', 'Do glenohumeral internal rotation measurements differ depending on stabilization techniques?', 'Does ephA2 Receptor Signaling mediate Inflammatory Responses in Lipopolysaccharide-Induced Lung Injury?',
                           'Do mitochondrial Gene Expression Profiles Are Associated with Maternal Psychosocial Stress in Pregnancy and Infant Temperament?', 'Does osteodystrophy in Cholestatic Liver Diseases be Attenuated by Anti-γ-Glutamyl Transpeptidase Antibody?', 
                           'Are yKL-40 levels independently associated with albuminuria in type 2 diabetes?', 'Does his bundle activate faster than ventricular myocardium during prolonged ventricular fibrillation?', 'Is phosphorylation of Shox2 required for its function to control sinoatrial node formation?', 
                           'Does acute Myocardial Infarction be a Risk Factor for New Onset Diabetes in Patients with Coronary Artery Disease?', 'Are cD44 and RHAMM essential for rapid growth of bladder cancer driven by loss of Glycogen Debranching Enzyme ( AGL )?', 
                           'Does the rgg0182 gene encode a transcriptional regulator required for the full Streptococcus thermophilus LMG18311 thermal adaptation?']

# for id in id_test_pm:
#     questions.append(get_instance_from_pubid(id)["question"])

# print(questions)