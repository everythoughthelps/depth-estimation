import pretrainedmodels


def resnext(groups, width_per_group):
    if groups == 64 and width_per_group == 4:
        return pretrainedmodels.resnext101_64x4d()
