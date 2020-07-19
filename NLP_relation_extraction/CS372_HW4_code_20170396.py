import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from allennlp.predictors.predictor import Predictor
import allennlp_models.syntax.constituency_parser
predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")

traindict={'activate':[["Here, we show that chronic treadmill exercise activates the mechanistic target of rapamycin (mTOR) pathway in mouse motor cortex.", ["treadmill exercise, activates, target"], "2019 Li Zhang Sci Adv. PMID: 31281888 Exercise Training Improves Motor Skill Learning via Selective Activation of mTOR"],
["Together, exercise activates mTOR pathway, which is necessary for spinogenesis, neuronal activation, and axonal myelination leading to improved motor learning.", ["exercise, activates, mTOR pathway"], "2019 Li Zhang Sci Adv. PMID: 31281888 Exercise Training Improves Motor Skill Learning via Selective Activation of mTOR"],
["In contrast, exposure to constant light, which perturbed the interval of inactivity (sleep) and led to the complete abolishment of activity/inactivity cycles, activated robustly proinflammatory state in the colon selectively via Stat3-dependent pathway.", ["exposure, activated, state"], "2017 Alena Sumova, Chronobiol Int. PMID: 29039977 Chronic Disruptions of Circadian Sleep Regulation Induce Specific Proinflammatory Responses in the Rat Colon"],
["Research has shown that the observation of another's movement activates the corresponding motor representation in the observer." ,["observation, activates, motor representation"],"Marcel Brass, 2018, J Exp Psychol Hum Percept Perform. PMID: 29154630 Automatic Imitation of Multiple Agents: Simultaneous or Random Representation?"],
["Canonical finger postures, as used in counting, activate number knowledge, but the exact mechanism for this priming effect is unclear.",["finger postures, activate, number knowledge"]," Lindemann O, Cogn Process, 2017 PMID: 28374126 Finger Posing Primes Number Comprehension"],
["LPS and numerous pore-forming exotoxins also activate the inflammasome, the molecular platform that allows the release of mature IL-1β and IL-18." ,["LPS and exotoxins, activate, inflammasome"], "Jean-Marc Cavaillon, 2018 Toxicon. PMID: 29056305 Exotoxins and Endotoxins: Inducers of Inflammatory Cytokines"],
["Interestingly, we demonstrate that this task-switch does not eliminate the inward current but instead activates an outward current.",["task-switch, activates, current"],"Weiss KR, 2018, J Neurosci. PMID: 29934354 Cellular Effects of Repetition Priming in the Aplysia Feeding Network Are Suppressed During a Task-Switch But Persist and Facilitate a Return to the Primed State"],
["Compared to HC, PD_OFF activated the bilateral putamen less, and this was compensated by higher activation of the anterior insula.",["PD_OFF, activated, putamen"],"Kathleen L Poston, 2018, Neuropsychologia. PMID: 30040957 Dopamine-related Dissociation of Cortical and Subcortical Brain Activations in Cognitively Unimpaired Parkinson's Disease Patients OFF and ON Medications"],
["Action observation activates brain areas involved in performing the same action and has been shown to increase motor learning, with potential implications for neurorehabilitation.",["Action observation, activates, brain areas"],"Gowen E, 2016, Exp Brain Res. PMID: 26892882 Enhancing Voluntary Imitation Through Attention and Motor Imagery"],
["Thus, RTKs allosterically activate PI3Kα; however, merging their action with Ras accomplishes full activation.",["RTKs, activate, PI3Kα"],"Hyunbum Jang, 2019, Front Oncol. PMID: 31799192 Does Ras Activate Raf and PI3K Allosterically?"],
["In the ectoderm, G protein-coupled receptors (GPCRs) and their downstream heterotrimeric G proteins (Gα and Gβγ) activate Rho1 both medial-apically, where it exhibits pulsed dynamics, and at junctions, where its activity is planar polarized.",["receptors and G proteins, activate, Rho1"],"Thomas Lecuit, 2019, Curries Biol PMID: 31522942 Distinct RhoGEFs Activate Apical and Junctional Contractility Under Control of G Proteins During Epithelial Morphogenesis"],
["Furthermore, Gβ13F/Gγ1 activates junctional Rho1 and exerts quantitative control over planar polarization of Rho1.",["Gβ13F/Gγ1, activates, Rho1"],"Thomas Lecuit,2019, Curries Biol PMID: 31522942 Distinct RhoGEFs Activate Apical and Junctional Contractility Under Control of G Proteins During Epithelial Morphogenesis"],
["The present study demonstrates that mammalian cannabinoid receptor ligands activate a conserved cannabinoid signaling system in C.elegans and also modulate monoaminergic signaling, potentially affecting an array of disorders, including anxiety and depression.",["receptor ligands, activate, system"],"Richard Komuniecki, 2017, J Neurosci PMID: 28188220 Cannabinoids Activate Monoaminergic Signaling to Modulate Key C. elegans Behaviors"],
["These DSBs in adaptive and innate immune cells activate the cDDR.", ["DSBs, activate, cDDR"], "Barry P Sleckman, 2019, Nat Rev Immunol. PMID: 30778174 At the Intersection of DNA Damage and Immune Responses"],
["Here, we report that OMVs isolated from the probiotic Escherichia coli Nissle 1917 and the commensal ECOR12 activate NOD1 signaling pathways in intestinal epithelial cells.",["OMVs, activate, NOD1"],"Laura Baldomà, 2018, Front Microbiol. PMID: 29616010 Outer Membrane Vesicles From Probiotic and Commensal Escherichia coli Activate NOD1-Mediated Immune Responses in Intestinal Epithelial Cells"],
["Using the T47D human breast cancer cell line, we have found that estradiol and progesterone synergistically activate HERV-K through nuclear receptors.",["estradiol and progesterone, activate, HERV-K"],"Yingguang Liu, 2019, AIDS Res Hum Retroviruses. PMID: 30565469 Female Sex Hormones Activate Human Endogenous Retrovirus Type K Through the OCT4 Transcription Factor in T47D Breast Cancer Cells"],
],
'inhibit':[["Coffee component 6, however, is a potent inhibitor of both Aβ and tau fibrillization, and also inhibits Aβ oligomerization (IC50 = 42.1 μM).",["Coffee component 6, inhibits, Aβ oligomerization"],"Donald F Weaver, 2018, Front Neurosci. PMID: 30369868, Phenylindanes in Brewed Coffee Inhibit Amyloid-Beta and Tau Aggregation"],
["Both of these molecular types can adsorb to droplet surfaces and inhibit lipid oxidation, but emulsifiers can also stabilize droplets against aggregation whereas coemulsifiers cannot.",["types, inhibit, lipid oxidation"],"Eric Decker, 2018, J Agric Food Chem. PMID: 29227097 Interfacial Antioxidants: A Review of Natural and Synthetic Emulsifiers and Coemulsifiers That Can Inhibit Lipid Oxidation"],
["Here we show that polysaccharides obtained from soil bacteria inhibit sucrose-induced hyperglycemia in an in vivo silkworm evaluation system.",["polysaccharides, inhibit, hyperglycemia"],"Sekimizu K. 2018, Drug Discov Ther. PMID: 30146618 Bacterial polysaccharides inhibit sucrose-induced hyperglycemia in silkworms."],
["Cephalosporins bind to the penicillin-binding proteins on bacteria and inhibit synthesis of the bacterial cell wall, causing cell lysis particularly in rapidly growing organisms.",["Cephalosporins, bind, proteins","Cephalosporins, inhibit, synthesis"],"No author, 2017, LiverTox PMID: 31643977 Cephalosporins."],
["Two E. coli effectors inhibit RPS3 nuclear translocation.",["Two E. coli effectors, inhibit, RPS3 translocation"],"Hardwidge PR. 2018, Pathogens. PMID: 30405005 SseL Deubiquitinates RPS3 to Inhibit Its Nuclear Translocation."],
["Recent research has shown that lithium (Li) can inhibit pro-inflammatory cytokine release in vitro via affecting the pharmacotherapy of psychiatric illnesses.",["lithium, inhibit, cytokine release"],"Chengtie Wu, 2018, J Mater Chem B. PMID: 32254931 Lithium-containing biomaterials inhibit osteoclastogenesis of macrophages in vitro and osteolysis in vivo."],
["Here we demonstrate that baicalein and baicalin can significantly inhibit human colon cancer cell growth and proliferation.",["baicalein and baicalin, inhibit, colon cancer cell growth and proliferation"],"Guangyong Peng, 2018, Oncotarget, PMID: 29732005 Baicalein and Baicalin Inhibit Colon Cancer Using Two Distinct Fashions of Apoptosis and Senescence"],
["This gene can promote proliferation and inhibit apoptosis in lung cancer.",["gene, inhibit, apoptosis"],"Dong Xu, 2019, Ann Clin Lab Sci. PMID: 31882431 Knockdown of PDRG1 Could Inhibit the Wnt Signaling Pathway in Esophageal Cancer Cells."],
["We conclude that CADs indeed inhibit adipocyte differentiation, as shown morphologically, at the level of lipid droplet formation and on the expression of genetic markers of adipocytes.",["CADs, inhibit, adipocyte differentiation"],"Munic Kos V. 2018, Eur J Pharmacol. PMID: 29627311 Lysosomotropic cationic amphiphilic drugs inhibit adipocyte differentiation in 3T3-L1K cells via accumulation in cells and phospholipid membranes, and inhibition of autophagy."],
["CTR and ECO could also inhibit S. mutans biofilm formation and reduce the viability of preformed biofilm.",["CTR and ECO, inhibit, S. mutans biofilm formation"],"Yuqing Li, 2017, Arch Oral Biol. PMID: 27764679 Clotrimazole and Econazole Inhibit Streptococcus Mutans Biofilm and Virulence in Vitro"],
["We further demonstrate that monocyclic glucose-mimicking iminosugars inhibit isolated glycoprotein and glycolipid processing enzymes and that this inhibition also occurs in primary cells treated with these drugs.",["iminosugars, inhibit, glycoprotein and processing enzymes"],"Nicole Zitzmann. 2016, PLoS Negl Trop Dis. PMID: 26974655 Iminosugars Inhibit Dengue Virus Production via Inhibition of ER Alpha-Glucosidases--Not Glycolipid Processing Enzymes."],
["CHML might inhibit the growth of U251 cells and promote apoptosis by up-regulating the expressions of Caspase-8 and Caspase-3; CHML also induced autophagy of U251 cells by promoting the expressions of MAP LC-3 and Beclin-1.",["CHML, inhibit, growth","CHML, induced, autophagy"],"Meiqing Lou, 2019, Pak J Pharm Sci. PMID: 31969291 Cytotropic heterogeneous molecular lipids inhibit the growth of glioma cells by inducing apoptosis and autophagy."],
["In esophageal cancer cells, SPINK5 overexpression can inhibit Wnt/β-catenin signaling pathway.",["SPINK5 overexpression, inhibit, Wnt/β-catenin signaling pathway"],"Ling-Ling Tao, 2019, Cancer Med. PMID: 30868765 A novel tumor suppressor SPINK5 targets Wnt/β-catenin signaling pathway in esophageal cancer."],
["In particular, recent studies found that polyphenols extracted from Hibiscus sabdariffa by organic solvents can inhibit melanoma cell growth.",["polyphenols, inhibit, melanoma cell growth"],"Yang CF. 2016, J Tradit Complement Med. PMID: 28053887 Components in aqueous Hibiscus rosa-sinensis flower extract inhibit in vitro melanoma cell growth."],
["The marine polycyclic-ether toxin gambierol and 1-butanol (n-alkanol) inhibit Shaker-type Kv channels by interfering with the gating machinery.",["polycyclic-ether toxin gambierol and 1-butanol, inhibit, Shaker-type Kv channels"], "Labro AJ. 2016, Toxicon. PMID: 27475861 Gambierol and n-alkanols inhibit Shaker Kv channel via distinct binding sites outside the K(+) pore."],
["Tan IIA can induce tumor cell death and inhibit tumor growth.",["Tan IIA, induce, tumor cell death","Tan IIA, inhibit, tumor growth"],"Yongchun Yu, 2019, Aging (Albany NY). PMID: 31711043 Tanshinone IIA Mediates SMAD7-YAP Interaction to Inhibit Liver Cancer Growth by Inactivating the Transforming Growth Factor Beta Signaling Pathway"],
],
'bind':[["All of these sulfonated molecules bind to PAP248-286 and alter the conformation of the peptide, even though they have various structures, sizes, and configurations.",["molecules, bind, PAP248-286"],"Jiajie Zhang, 2018, ChemistryOpen. PMID: 29928568 Sulfonated Compounds Bind With Prostatic Acid Phosphatase (PAP 248-286) to Inhibit the Formation of Amyloid Fibrils"],
["Ternary complexes of the final step in extracellular signaling show how TGF-β GF dimers bind type I and type II receptors on the cell surface, and enable understanding of much of the specificity and promiscuity in extracellular signaling.",["TGF-β GF dimers, bind, type I and type II receptors"],"Timothy A Springer, 2016, Cold Spring Harb Perspect Biol. PMID: 27638177 Structural Biology and Evolution of the TGF-β Family"],
["Monoclonal antibodies produced by the HIC1-2B4 and HIC0-4F9 mouse hybridomas bind distinct surface molecules expressed on endocrine cells and have been validated for a number of experimental methods including immunohistochemistry and live cell sorting by flow cytometry.",["antibodies, bind, surface molecules"],"Philip R Streeter, 2018, J Immunol Methods. PMID: 29758224 Humanised Recombinant Antibody Fragments Bind Human Pancreatic Islet Cells"],
["Progestins used in endocrine therapies bind to multiple steroid receptors and are associated with several side-effects.",["Progestins, bind, receptors"],"Donita Africander, 2017, Biochem Biophys Res Commun. PMID: 28711501 Comparing the Androgenic and Estrogenic Properties of Progestins Used in Contraception and Hormone Therapy"],
["Pentatricopeptide repeat (PPR) proteins bind RNA via a mechanism that facilitates the customization of sequence specificity.",["Pentatricopeptide repeat proteins, bind, RNA"],"Alice Barkan, 2019, Plant Cell. PMID: 31123048 Ribonucleoprotein Capture by in Vivo Expression of a Designer Pentatricopeptide Repeat Protein in Arabidopsis"],
["We demonstrate here that artificial PPR proteins built from consensus PPR motifs selectively bind the intended RNA in vivo, and we use this property to develop a new tool for ribonucleoprotein characterization.",["PPR proteins, bind, RNA"],"Alice Barkan, 2019, Plant Cell. PMID: 31123048 Ribonucleoprotein Capture by in Vivo Expression of a Designer Pentatricopeptide Repeat Protein in Arabidopsis"],
["Also noteworthy is that very few proteins bind their own mRNAs that are not associated with ribosome function.",["proteins, bind, mRNAs"],"Meredith Root-Bernstein, 2016, J Theor Biol. PMID: 26953650 The Ribosome as a Missing Link in Prebiotic Evolution II: Ribosomes Encode Ribosomal Proteins That Bind to Common Regions of Their Own mRNAs and rRNAs"],
["Specific trans-acting factors, including 14-3-3, bind eB.",["factors, bind, eB"],"Joanna Floros, 2015, Am J Physiol Lung Cell Mol Physiol. PMID: 26001776 14-3-3 Isoforms Bind Directly Exon B of the 5'-UTR of Human Surfactant Protein A2 mRNA"],
["1) eB RNA pulldown assays showed that 14-3-3 isoforms specifically bind eB.",["14-3-3 isoforms, bind, eB"],"Joanna Floros, 2015, Am J Physiol Lung Cell Mol Physiol. PMID: 26001776 14-3-3 Isoforms Bind Directly Exon B of the 5'-UTR of Human Surfactant Protein A2 mRNA"],
["Since PS-ASOs bind to major P-body components, PS-ASOs may serve as scaffolds for P-body formation.",["PS-ASOs, bind, P-body components"],"Stanley T Crooke 2019, Nucleic Acid Ther. PMID: 31429620 Phosphorothioate Antisense Oligonucleotides Bind P-Body Proteins and Mediate P-Body Assembly"],
["Type I inhibitors bind to the active protein kinase conformation (DFG-Asp in, αC-helix in).",["Type I inhibitors, bind, protein kinase conformation"],"Robert Roskoski Jr 2016, Pharmacol Res. PMID: 26529477 Classification of Small Molecule Protein Kinase Inhibitors Based Upon the Structures of Their Drug-Enzyme Complexes"],
["In contrast, type VI inhibitors bind covalently to their target enzyme.",["type VI inhibitors, bind, target enzyme"],"Robert Roskoski Jr 2016, Pharmacol Res. PMID: 26529477 Classification of Small Molecule Protein Kinase Inhibitors Based Upon the Structures of Their Drug-Enzyme Complexes"],
["Most of these proteins bind to heparin in vitro, a highly sulfated GAG species, although heparan sulfate and/or chondroitin/dermatan sulfate are more frequent physiological ligands.",["proteins, bind, heparin"],"Ulf Lindahl 2018, Curr Opin Struct Biol. PMID: 29455055 Specificity of Glycosaminoglycan-Protein Interactions"],
["Many transcription factors preferentially bind close to the end of nucleosomal DNA, or to periodic positions on the solvent-exposed side of the DNA.",["transcription factors, bind, end or positions"],"Jussi Taipale, 2018, Nature. PMID: 30250250 The Interaction Landscape Between Transcription Factors and the Nucleosome"],
["In addition, several transcription factors usually bind to nucleosomal DNA in a particular orientation.",["transcription factors, bind, DNA"],"Jussi Taipale, 2018, Nature. PMID: 30250250 The Interaction Landscape Between Transcription Factors and the Nucleosome"],
["Several CWA proteins comprise modules that have different functions, and some individual domains can bind different ligands, sometimes by different mechanisms.",["domains, bind, ligands"],"Timothy J Foster, 2019,  Microbiol Spectr. PMID: 31267926 Surface Proteins of Staphylococcus aureus"],
],
'induce':[["Several of these analogs can induce SA-mediated defense and inhibit growth of Pseudomonas syringae in Arabidopsis.",["analogs, induce, defense","analogs, inhibit, growth"],"Zheng Qing Fu 2019, Int J Mol Sci. PMID: 31288496 Novel Salicylic Acid Analogs Induce a Potent Defense Response in Arabidopsis"],
["These analogs, when sprayed on Arabidopsis, can induce the accumulation of the master regulator of plant defense NPR1.",["analogs, induce, accumulation"],"Zheng Qing Fu 2019, Int J Mol Sci. PMID: 31288496 Novel Salicylic Acid Analogs Induce a Potent Defense Response in Arabidopsis"],
["In recent years, many studies have demonstrated that CAFs induce metastasis and drug resistance in cancer cells via exosomes.",["CAFs, induce, metastasis and drug resistance"],"C P Hu, 2019, QJM. PMID: 31106370 Snail1-dependent Cancer-Associated Fibroblasts Induce Epithelial-Mesenchymal Transition in Lung Cancer Cells via Exosomes"],
["Therefore, apelin may induce the progression of diabetic nephropathy by counteracting the myogenic response in smooth muscle cells.",["apelin, induce, progression"],"Xiangjun Zeng, 2018, FASEB J. PMID: 29522374 Apelin Impairs Myogenic Response to Induce Diabetic Nephropathy in Mice"],
["ICIs typically induce oral lichenoid reactions and xerostomia.",["ICIs, induce, reactions and xerostomia"],"Vincent Sibaud, 2018, Am J Clin Dermatol. PMID: 30374901 Toxic Side Effects of Targeted Therapies and Immunotherapies Affecting the Skin, Oral Mucosa, Hair, and Nails"],
["Targeted therapies and endocrine therapy also commonly induce alopecia, although this is still underreported with the latter.",["therapies and endocrine therapy, induce, alopecia"],"Vincent Sibaud, 2018, Am J Clin Dermatol. PMID: 30374901 Toxic Side Effects of Targeted Therapies and Immunotherapies Affecting the Skin, Oral Mucosa, Hair, and Nails"],
["Our findings demonstrate that hypoxic exosomes induce differential gene expression in recipient glioma cells.",["exosomes, induce, gene expression"],"Robert J Griffin, 2018, Biochem Biophys Rep. PMID: 29872742 Hypoxia-derived Exosomes Induce Putative Altered Pathways in Biosynthesis and Ion Regulatory Channels in Glioblastoma Cells"],
["Recent clinical studies in stroke and traumatic brain injury (TBI) victims suffering chronic neurological injury present evidence that hyperbaric oxygen therapy (HBOT) can induce neuroplasticity.",["oxygen therapy, induce, neuroplasticity"],"Shai Efrati, 2017, Front Hum Neurosci. PMID: 29097988 Hyperbaric Oxygen Therapy Can Induce Angiogenesis and Regeneration of Nerve Fibers in Traumatic Brain Injury Patients"],
["Spicy foods might induce heartburn, but the exact mechanism is not known.",["foods, induce, heartburn"], "Dan L Dumitrascu, 2019, Curr Med Chem. PMID: 28521699 Food and Gastroesophageal Reflux Disease"],
["Beer and wine induce gastroesophageal reflux, mainly in the first hour after intake.",["Beer and wine, induce, reflux"],"Dan L Dumitrascu, 2019, Curr Med Chem. PMID: 28521699 Food and Gastroesophageal Reflux Disease"],
["Unlike the flat protein films, the unique protein nanoridges can induce the differentiation of human mesenchymal stem cells (MSCs) into osteoblasts without any additional inducers, as well as the formation of bone tissue in a subcutaneous rat model even when not seeded with MSCs.",["protein nanoridges, induce, differentiation"],"Chuanbin Mao, 2017, Adv Funct Mater. PMID: 29657571 Ice-Templated Protein Nanoridges Induce Bone Tissue Formation"],
["Moreover, the nanoridged films induce less inflammatory infiltration than the flat films in vivo.",["films, induce, infiltration"],"Chuanbin Mao, 2017, Adv Funct Mater. PMID: 29657571 Ice-Templated Protein Nanoridges Induce Bone Tissue Formation"],
["Environmental pollutants and allergens induce oxidative stress and mitochondrial dysfunction, leading to key features of allergic asthma.",["pollutants and allergens, induce, stress and dysfunction"],"Peisong Gao, 2019, Front Immunol. PMID: 31849968 Environmental Exposures and Asthma Development: Autophagy, Mitophagy, and Cellular Senescence"],
["Additionally, mitochondrial dysfunction can induce cell senescence due to excessive ROS production, which affects airway diseases.",["dysfunction, induce, cell senescence"],"Peisong Gao, 2019, Front Immunol. PMID: 31849968 Environmental Exposures and Asthma Development: Autophagy, Mitophagy, and Cellular Senescence"],
["Lipid-modifying agents may also induce hyperglycaemia, and the diabetogenic effect seems to differ between the different types and daily doses of statins.",["agents, induce, hyperglycaemia"],"Chaker Ben Salem, 2015,  Drug Saf. PMID: 26370106 Drug-Induced Hyperglycaemia and Diabetes"],
["Ketoacidosis may occur in patients receiving beta-adrenergic stimulants, and theophylline may also induce hyperglycaemia.",["theophylline, induce, hyperglycaemia"],"Chaker Ben Salem, 2015,  Drug Saf. PMID: 26370106 Drug-Induced Hyperglycaemia and Diabetes"],
],
'abolish':[["Collectively, GRg1 or DXM treatment significantly abolishes IMQ-induced psoriasis-like dermatitis by lowering PASI score, inflammation through downregulating NF-κB signaling pathway.",["GRg1 or DXM treatment, abolishes, dermatitis"],"Bo Zhang, 2019, J Food Biochem. PMID: 31502279 Ginsenoside Rg1 Abolish Imiquimod-Induced Psoriasis-Like Dermatitis in BALB/c Mice via Downregulating NF-κB Signaling Pathway"],
["NEK7 knockdown abolish ATP + LPS-induced pyroptosis in vitro and improved DSS-induced chronic colitis in vivo.",["NEK7 knockdown, abolish, pyroptosis"],"Lianwen Yuan, 2019, Cell Death Dis. PMID: 31787755 NEK7 Interacts With NLRP3 to Modulate the Pyroptosis in Inflammatory Bowel Disease via NF-κB Signaling"],
["The demonstration here that anterior thalamic nuclei lesions abolish latent inhibition is consistent with emerging evidence of the importance of these thalamic nuclei for attentional control.",["nuclei lesions, abolish, inhibition"],"John P Aggleton, 2018, Behav Neurosci. PMID: 30321027 Anterior Thalamic Nuclei, but Not Retrosplenial Cortex, Lesions Abolish Latent Inhibition in Rats"],
["However, recent research suggests that prolonged sedentary behavior might abolish these healthy metabolic benefits.",["behavior, abolish, benefits"],"Edward F Coyle, 2019, J Appl Physiol (1985). PMID: 30763169 Inactivity Induces Resistance to the Metabolic Benefits Following Acute Exercise"],
["The cumulative data in nonclinical settings suggest that cross-education can completely abolish expected declines in strength and muscle size in the range of ∼13% and ∼4%, respectively, after 3-4 weeks of immobilization of a healthy arm.",["cross-education, abolish, declines"], "Jonathan P Farthing, 2018, Appl Physiol Nutr Metab. PMID: 29800529 Contralateral Effects of Unilateral Training: Sparing of Muscle Strength and Size After Immobilization"],
["Proteins can abolish these RNA structures through binding to one of the complementary strands.",["Proteins, abolish, RNA structures"],"Stefan Stamm, 2019, Biochim Biophys Acta Gene Regul Mech. PMID: 31421281 Pre-mRNA Structures Forming Circular RNAs"],
["Droperidol used as prophylaxis for postoperative nausea abolishes TcMEPs.",["Droperidol, abolishes, TcMEPs"],"Jose Ángel Torres Dios, 2017, Turk J Anaesthesiol Reanim. PMID: 28377841 What Can We Learn From Two Consecutive Cases? Droperidol May Abolish TcMEPs"],
["These mutations abolish or reduce CST interaction with RAD51, disrupt RAD51 foci formation, and/or diminish binding to GC-rich genomic fragile sites under replication stress.",["mutations, abolish, CST interaction"],"Weihang Chai, 2018, Nucleic Acids Res. PMID: 29481669 Pathogenic CTC1 Mutations Cause Global Genome Instabilities Under Replication Stress"],
["More importantly, the addition of 100 μM NPPB could abolish the promoting effect of Gata4 silencing on PTM cell contraction.",["addition, abolish, effect"],"Yi-Xun Liu, 2018, Reproduction. PMID: 30306767 GATA4 Is a Negative Regulator of Contractility in Mouse Testicular Peritubular Myoid Cells"],
["Deletion of VEGFA can abolish IL-6 induced progression of HA, suggesting the essential role of VEGFA in IL-6 induced HA development.",["Deletion, abolish, progression"],"Ji Yuan, 2018, Eur J Pharmacol. PMID: 30342949 Interleukin-6 (IL-6) Triggers the Malignancy of Hemangioma Cells via Activation of HIF-1α/VEGFA Signals"],
["STAT3 inhibitor CPA7 or si-STAT3 can abolish IL-6 induced upregulation of HIF-1α in HDEC cells.",["STAT3 inhibitor CPA7 or si-STAT3, abolish, upregulation"],"Ji Yuan, 2018, Eur J Pharmacol. PMID: 30342949 Interleukin-6 (IL-6) Triggers the Malignancy of Hemangioma Cells via Activation of HIF-1α/VEGFA Signals"],
["Elimination of Lamin A/C can abolish p53-induced p16 expression and BMI-1/MEL-18 reduction.",["Elimination, abolish, p16 expression and BMI-1/MEL-18 reduction"], "Bum-Joon Park, 2019, Cell Death Dis. PMID: 30728349 p53 Induces Senescence Through Lamin A/C Stabilization-Mediated Nuclear Deformation"],
["In addition, we revealed that miR-27a-3p mimics could abolish the effects of FOXD2-AS1 overexpression on cell proliferation, inflammation, and ECM degradation in chondrocytes.",["mimics, abolish, effects"]," Shuogui Xu, 2019, Artif Cells Nanomed Biotechnol. PMID: 30945573 LncRNA FOXD2-AS1 Induces Chondrocyte Proliferation Through Sponging miR-27a-3p in Osteoarthritis"],
["Mutations of these potential iron binding sites at the N-terminus, as well as a likely iron binding site at the C-terminus of dZIP13, completely abolish the iron-dependent upregulation in the yeast and the fruit fly.",["Mutations, abolish, upregulation"],"Bing Zhou, 2019, Biochim Biophys Acta Mol Cell Res. PMID: 31229649 Drosophila ZIP13 Is Posttranslationally Regulated by Iron-Mediated Stabilization"],
["These mutations also completely abolish FLNA's interactions with protein tyrosine phosphatase nonreceptor type 12, which has been suggested to contribute to the pathogenesis of FLNA-MVD.",["mutations, abolish, interactions"],"Ulla Pentikäinen, 2019, Biophys J. PMID: 31542223 Critical Structural Defects Explain Filamin A Mutations Causing Mitral Valve Dysplasia"],
["Androgenic metabolites can abolish the growth of DMBA-tumors and prevent the appearance of new tumors.",["metabolites, abolish, growth"],"F Heinrich Wieland, 2019, Cell Death Dis. PMID: 31235695 The Beneficial Androgenic Action of Steroidal Aromatase Inactivators in Estrogen-Dependent Breast Cancer After Failure of Nonsteroidal Drugs"],
]}

testdict={'activate':[["Optimized compounds activate nucleotide exchange at single-digit micromolar concentrations in vitro.",["compounds, activate, nucleotide exchange"],"Stephen W Fesik, 2018, ACS Med Chem Lett. PMID: 30258545 Discovery of Quinazolines That Activate SOS1-Mediated Nucleotide Exchange on RAS"],
["Smells influence and modify the hedonic qualities of eating experience, and in contrast to smells not associated with food, perception of food-associated odors may activate dopaminergic brain areas.",["perception, activate, brain areas"],"Thomas Hummel, 2017, Front Hum Neurosci. PMID: 29311879 Food-Related Odors Activate Dopaminergic Brain Areas"],
["They show that stator units, which normally interact with the flagellum to power rotation, can alternatively interact with and activate an enzyme that synthesizes cyclic di-GMP in Pseudomonas aeruginosa.",["stator units, activate, enzyme"],"Daniel B Kearns, 2019, J Bacteriol. PMID: 30962352 Flagellar Stators Activate a Diguanylate Cyclase To Inhibit Flagellar Stators"],
["Many odors activate the intranasal chemosensory trigeminal system where they produce cooling and other somatic sensations such as tingling, burning, or stinging.",["odors, activate, system"], "Hummel T, Frasnelli J. 2019, Handb Clin Neurol. PMID: 31604542 The intranasal trigeminal system."],
],
'inhibit':[["Boswellia preparations inhibit 5-lipoxygenase and prevent the release of leukotrienes, thus having an anti-inflammatory effect in ulcerative colitis, irritable bowel syndrome, bronchitis and sinusitis.",["Boswellia preparations, inhibit, 5-lipoxygenase"],"Bożena Kiczorowska, 2016, Postepy Hig Med Dosw, PMID: 27117114 Frankincense--therapeutic Properties"],
["NleH1 inhibits RPS3 phosphorylation by IKK-β, an essential aspect of the RPS3 nuclear translocation process.",["NleH1, inhibits, RPS3 phosphorylation"],"Philip R Hardwidge, 2018, Pathogens. PMID: 30405005 SseL Deubiquitinates RPS3 to Inhibit Its Nuclear Translocation"],
["All instant coffee extracts inhibit fibrillization of Aβ and tau, and promote α-synuclein oligomerization at concentrations above 100 μg/mL.",["coffee extracts, inhibit, fibrillization"],"Donald F Weaver, 2018, Front Neurosci. PMID: 30369868 Phenylindanes in Brewed Coffee Inhibit Amyloid-Beta and Tau Aggregation"],
["Taken together, mesenchymal stem cell-derived exosomal miR-199a can inhibit the progression of glioma by down-regulating AGAP2.",["miR-199a, inhibit, progression"],"Binghui Qiu, 2019, Aging (Albany NY). PMID: 31386624 Exosomes Derived From microRNA-199a-overexpressing Mesenchymal Stem Cells Inhibit Glioma Progression by Down-Regulating AGAP2"],
],
'bind':[["Recently, emerging evidence demonstrated that some TFs could bind to DNA motifs containing highly methylated CpGs both in vitro and in vivo.",["TFs, bind, DNA motifs"],"Yadong Wang 2018, Nucleic Acids Res. PMID: 29145608 MeDReaders: A Database for Transcription Factors That Bind to Methylated DNA"],
["Most Abs established as markers for autoimmune disease bind to cytoplasmic or nuclear substances.",["Abs, bind, substances"],"G Kurosawa 2018, Biochem Biophys Res Commun. PMID: 29944883 Isolation of Human Monoclonal Antibodies That Bind to Two Different Antigens and Are Encoded by Germline V H and V L Genes"],
["Here we show, using a novel method, that the S1A domain specifically binds to the nasal epithelium of dromedary camels, alveolar epithelium of humans, and intestinal epithelium of common pipistrelle bats.",["S1A domain, binds, epithelium, alveolar epithelium, and intestinal epithelium"],"Bart L Haagmans 2019, J Virol. PMID: 31167913 Species-Specific Colocalization of Middle East Respiratory Syndrome Coronavirus Attachment and Entry Receptors"],
["Yet other GAG epitopes bind protein ligands with intermediate specificity and variable affinity.",["GAG epitopes, bind, protein ligands"], "Ulf Lindahl 2018, Curr Opin Struct Biol. PMID: 29455055 Specificity of Glycosaminoglycan-Protein Interactions"],
],
'induce':[["Based on the Naranjo Adverse Drug Reaction scale, it is probable that the eosinophilia was induced by pregabalin, as the Naranjo probability score was calculated to be 8.",["eosinophilia, was induced by, pregabalin"],"C M Neethu, 2018, Consult Pharm. PMID: 29880093 A Case Report on Pregabalin-Induced Eosinophilia"],
["Oncogenes induce premature S phase, resulting in replication-transcription conflicts and replication stress.",["Oncogenes, induce, S phase"],"No author, 2018, Cancer Discov. PMID: 29500300 Oncogenes Induce Replication Stress via Intragenic Replication Origins"],
["It is widely observed that antimicrobials can induce pseudo-allergic reactions (i.e. IgE-independent mechanism) with symptoms ranging from skin inflammation to life-threatening systemic anaphylaxis.",["antimicrobials, induce, reactions"],"Langchong He, 2017, Eur J Immunol. PMID: 28688196 Typical Antimicrobials Induce Mast Cell Degranulation and Anaphylactoid Reactions via MRGPRX2 and Its Murine Homologue MRGPRB2"],
["We conclude that 4-substituted phenols can induce specific T-cell responses against melanocytes and melanoma cells, also acting at distant, unexposed body sites, and may confer a risk of chemical vitiligo.",["phenols, induce, T-cell responses"],"Rosalie M Luiten, 2019, Pigment Cell Melanoma Res. PMID: 30767390 Mechanism of Action of 4-substituted Phenols to Induce Vitiligo and Antimelanoma Immunity"],
],
'abolish':[["In this study, we have confirmed that RBM10 decreases the activation of RAP1 and found that EPAC stimulation and inhibition can abolish the effects of RBM10 knockdown and overexpression, respectively, and regulate cell growth.",["EPAC stimulation and inhibition, abolish, effects"],"Ke Wang, 2019, Cell Mol Med. PMID: 30955253 RBM10 Inhibits Cell Proliferation of Lung Adenocarcinoma via RAP1/AKT/CREB Signalling Pathway"],
["However, neutralizing antibodies (NAbs) against AAV capsids can abolish AAV infectivity on target cells, reducing the transduction efficacy.",["antibodies, abolish, AAV infectivity"],"Weidong Xiao, 2018, Mol Ther Methods Clin Dev. PMID: 30623003 Rapid AAV-Neutralizing Antibody Determination With a Cell-Binding Assay"],
["Both endotracheal intubation and atropine administration in CPR process can abolish these waves.",["intubation and atropine administration, abolish, waves"],"Ahmad Reza Dehpour, 2019, Med Hypotheses PMID: 31010488 Brain Wave Disturbance and Cognitive Impairment After CPR"],
["Anti-inflammatory agents can improve myofibroblastic transdifferentiation and abolish TLT formation, suggesting that targeting these inflammatory fibroblasts can potentially ameliorate kidney disease.",["agents, abolish, TLT formation"],"Motoko Yanagita, 2017, Inflamm Regen. PMID: 29259716 Resident Fibroblasts in the Kidney: A Major Driver of Fibrosis and Inflammation"],
],
}

grammar = r"""
    PARA: {<PARA><FW|CD|NN.*|JJ.*|VB.*|DT|IN|CC>*<PARA>}
    NP : {(<JJ.*|DT><IN>)?<DT>?<VBG|VBN><JJ.*|NN.*|VBG>+(<CC|SYM><DT|JJ.*|FW|PARA|CD|PRP.*|NN.*>+)?} 
         {(<JJ.*|DT><IN>)?<DT|JJ.*|FW|PARA|CD|PRP.*|POS|NN.*>+(<CC|SYM><DT|JJ.*|FW|PARA|CD|PRP.*|NN.*>+)?}
    PP : {<IN><CD>?<NP>}
    VP : {<MD|VBD|VBZ>?<RB>?<VB.*>(<CC><VB.*>)?<TO|NP|PP|RB.*>+(<CC>+<TO|NP|PP|RB.*>)?}
    CLAUSE : {<NP>+<PP>?<VP>+(<CC><VP>+)?}
    EQUAL : {<,><VP|NP|IN|CC>+<,>}
"""
cp = nltk.RegexpParser(grammar)

stemmer = PorterStemmer()

# five target verb 
targetverblist=['inhibit', 'induc', 'activ', 'abolish', 'bind']

resulttuplist=[]

def getVerb(parent,level,beforenplist,result,verbflag):
    for node in parent:
        if type(node) is nltk.Tree:
            if node.label() == 'NP':
                # third entry
                if verbflag==1:
                    afternounflag=0
                    for index, afternp in enumerate(node):
                        if (('NN' in afternp[1]) or ('FW' in afternp[1]) or ('CD' in afternp[1])) :
                            if type(afternp) is nltk.Tree:
                                continue
                            if afternounflag==0:
                                result.append(afternp[0])
                                afternounflag=1
                            else:
                                result[2]+=' '+afternp[0]
                            if index<len(node)-1 and node[index+1][1]=='CC':
                                result[2]+=' '+node[index+1][0] 
                    break
                # first entry
                beforenplist.append((node,level,parent))
            elif node.label() == 'PP':
                if node[0][0]!='to' and node[0][0]!='by':
                    continue
                # passive sent
                if node[0][0]=='by':
                    if parent.label()=='VP':
                        for child in parent:
                            if ('VB' in child[1]) and (stemmer.stem(child[0]) in targetverblist):
                                childindex=parent.index(child)
                                if childindex!=0:
                                    expectbe=parent[childindex-1]
                                    if 'VB' in expectbe[1]:
                                        result[1]=expectbe[0]+' '+result[1]+' '+node[0][0]
            #recursive tree traversal
            getVerb(node,level+1,beforenplist,result,verbflag)
        else:
            if ('VB' in node[1]) and (stemmer.stem(node[0]) in targetverblist) and parent.label()=='VP':
                for beforenp in beforenplist:
                    # height differ by 1 and VP has to be sibling with NP
                    if beforenp[1]+1==level and parent in beforenp[2]:
                        for index, nounphrase in enumerate(beforenp[0]):
                            if (('NN' in nounphrase[1]) or ('FW' in nounphrase[1]) or ('CD' in nounphrase[1])):
                                # first entry (noun)
                                if type(nounphrase) is nltk.Tree:
                                    continue
                                if verbflag==1:
                                    result[0]+=' '+nounphrase[0]
                                else:
                                    result.append(nounphrase[0])
                                    # second entry (target verb)
                                    result.append(node[0])
                                    verbflag=1 
                                # noun connected with CC word
                                if index<len(beforenp[0])-1 and beforenp[0][index+1][1]=='CC':
                                    result[0]+=' '+beforenp[0][index+1][0]                 
    return result
                    

def chunking(dataset):
    for sent in dataset:
        for entry in dataset[sent]:
            examplesent=entry[0]
            result=predictor.predict_json({"sentence":examplesent})
            tagged=[]
            for index in range(len(result['tokens'])):
                tagged.append((result['tokens'][index],result['pos_tags'][index]))
            for index, pair in enumerate(tagged):
                # words including hyph and sym should have 1 pos tag
                while tagged[index][1]=='HYPH' or tagged[index][0]=='-' or tagged[index][1]=='SYM':
                    tagged[index+1]=(tagged[index-1][0]+tagged[index][0]+tagged[index+1][0],tagged[index+1][1])
                    tagged.pop(index)
                    tagged.pop(index-1)
                    if index>len(tagged)-1:
                        break
                if pair[1]=='POS':
                    tagged[index]=(tagged[index-1][0]+tagged[index][0],tagged[index][1])
                    tagged.pop(index-1)
            # special tag for word 'that'
            thattag=('that','THAT')
            # special tag for parenthesis
            openparen=('(','PARA')
            closeparen=(')','PARA')
            thattagged = [thattag if thattag[0] == e[0] else e for e in tagged]
            opentagged = [openparen if openparen[0] == e[0] else e for e in thattagged]
            closetagged = [closeparen if closeparen[0] == e[0] else e for e in opentagged]

            # last ended with other tag (ex)eB. -> NN tag)
            dotexpected=closetagged[len(closetagged)-1]
            if dotexpected[1]!='.':
                closetagged[len(closetagged)-1]=(dotexpected[0][:-1],dotexpected[1])
                closetagged.append(('.','.'))
            parsedsent=cp.parse(closetagged)
            resulttuple=getVerb(parsedsent,0,[],[],0)
            # tagger backoff
            if len(resulttuple)==0:
                for index, pair in enumerate(closetagged):
                    if stemmer.stem(pair[0]) in targetverblist:
                        closetagged[index]=(pair[0],'VB')
                        parsedsent=cp.parse(closetagged)
                        resulttuple=getVerb(parsedsent,0,[],[],0)
            resultvallist=[]
            # group into three words
            while(len(resulttuple)>3):
                returntup= (', '.join(resulttuple[:3]))
                resultvallist.append(returntup)
                resulttuple=resulttuple[3:]
            resultvallist.append(', '.join(resulttuple))
            resulttuplist.append(resultvallist)

def evaluate(dataset,resulttuplist):
    tupindex=0
    true_positive=0
    false_positive=0
    false_negative=0
    for sent in dataset:
        for entry in dataset[sent]:
            expectedlist=entry[1]
            resultlist=resulttuplist[tupindex]
                
            for resulttup in resultlist:
                if resulttup in expectedlist:
                    # correct -> TP
                    true_positive+=1
                else:
                    # false, wrong -> FP
                    false_positive+=1
            for expecttuple in expectedlist:
                if expecttuple not in resultlist:
                    # missed -> FN
                    false_negative+=1
            tupindex+=1
    precision=(true_positive)/(true_positive+false_positive)
    print("Precision: ",precision)
    recall=(true_positive)/(true_positive+false_negative)
    print("Recall: ",recall)
    if(precision+recall==0):
        f_score=0
    else:
        f_score=2*(precision*recall)/(precision+recall)
    print("F score: ",f_score)


senttxt=open('CS372_HW4_input_20170396.txt','w',newline="")

# train data
chunking(traindict)
evaluate(traindict,resulttuplist)

tuplistindex=0
for sent in traindict:
    for entry in traindict[sent]:
        senttxt.write(str(entry+resulttuplist[tuplistindex]+['training data'])+'\n')
        tuplistindex+=1

# test data
resulttuplist.clear()
chunking(testdict)
evaluate(testdict,resulttuplist)

tuplistindex=0
for sent in testdict:
    for entry in testdict[sent]:
        senttxt.write(str(entry+resulttuplist[tuplistindex])+'\n')
        tuplistindex+=1
    
senttxt.close()