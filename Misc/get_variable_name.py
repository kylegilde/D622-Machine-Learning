def get_variable_name(**the_variable):
    """
    :param input is in the form of the_variable=the_variable
    :return: the variable's name as a string
    :credit: https://stackoverflow.com/a/19201952/4463701
    """
    return [x for x in the_variable][0]