def show(dictionary, number=0):
    """
    return an element of a dictionary
    If number is not specified, returns the values associated with the first key
    """
    try:
        return(dictionary[list(dictionary.keys())[number]])
    except:
        print("something's wrong")

def subset(dictionary, filter_dict):
    """
    Return a subset of a dictionary, specified in filter_dict (itself a dictionary)
    filter_dict is {attrib:["attrib_value_x","attrib_value_y",..]}, where 
        attrib is an attribute of the elements of dictionary, or a metadata attrib and attrib_value is a list
        of the values of such attrib that the elements of returned dictionary can have    
    """
    if type(dictionary) != type(dict()):
        print("subset function error: type dictionary should be dict")
        return
    if type(filter_dict) != type(dict()):
        print("subset function error: type filter_dict should be dict")
        return
    return_dict = dictionary
    for i, j in filter_dict.items():
        a = {}
        for key,value in return_dict.items():
            if hasattr(value,'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                try:
                    if value.__getattr__('meta')[i] in j:
                        a[key] = value
                except:
                    pass
            else:
                try:
                    if value.__getattr__(i) in j:
                        a[key] = value
                except:
                    pass
        return_dict = a

    return return_dict

def set_attrib(dictionary, attribute):
    """
    returns the set of attribute values for dictionary
    """
    return_set = set()
    for i in dictionary.values():
        if hasattr(i, 'meta') & (type(i.meta) == type({})) & (attribute in i.meta.keys()):
            try:
                return_set.add(i.__getattr__('meta')[attribute])
            except:
                pass
        else:
            try:
                return_set.add(i.__getattr__(attribute))
            except:
                pass
    
    return return_set 