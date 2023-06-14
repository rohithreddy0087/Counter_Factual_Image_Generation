def calculate_proximity(imgs, imgs_cf):
    """
        calculates the proximity score.
    :param imgs:
    :param imgs_cf:
    :return:
    """
    n, c, h, w = imgs.size()
    summ = abs(imgs - imgs_cf).sum().item()
    return summ/(n*c*h*w)


def calculate_validity(pred , target):
    """
    calcuates the validity score.
    :param classifier:
    :param imgs:
    :param target_cls:
    :return:
    """
    validity = pred.eq(target.view_as(pred)).sum().item()
    return validity