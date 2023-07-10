# 0.984
import Orange

base = Orange.data.Table("./data/credit_data_regras.csv")

base_test , base_train = Orange.evaluation.testing.sample(base, n=0.25)

cn2 = Orange.classification.rules.CN2Learner()
regras = cn2(base_train)

for rule in regras.rule_list:
    print(rule)

predicts = Orange.evaluation.testing.TestOnTestData(
    base_train, base_test, [lambda testdata: regras]
)

Orange.evaluation.CA(predicts)
