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


MAIN_DIR_PATH = get_main_dir(1) # nopep8

from utils.lecture_xml import pmid_to_pmcid

id_questions_pm =  [27138702, 17624920, 12653172, 19067803, 19752385,20525838, 25636253,23321169, 19038998, 12417548, 9177511, 15466627, 16421117, 25416273, 21826766, 18041713, 22436018, 27752752, 8933126, 22897785, 23317773, 19479018, 16462004, 23441106, 27107135, 24463566, 21604079, 15942901, 20708103, 12844360, 14634566, 23712332, 15451917, 25375175, 10782911, 23456233, 25301361, 22356843, 17106929, 19455129, 15569302, 21981946, 26142205, 19096133, 20388204, 21827577, 23133587, 22351855, 15480106, 27492328]

id_question_pmc = []
for id in id_questions_pm:
    id_pmc = pmid_to_pmcid(id)
    id_question_pmc.append(id_pmc)

print(id_question_pmc)