import os as os
import uuid as uuid

def show(dictionary, number=0):
    """
    return an element of a dictionary
    If number is not specified, returns the values associated with the first key
    """
    try:
        return(dictionary[list(dictionary.keys())[number]])
    except:
        print("something's wrong")

def subset(dictionary, filter_dict, filter_style='all'):
    """
    Return a subset of a dictionary, specified in filter_dict (itself a dictionary)
    filter_dict is {attrib:["attrib_value_x","attrib_value_y",..]}, where 
        attrib is an attribute of the elements of dictionary, and attrib_value is a list
        of the values of such attrib that the elements of returned dictionary can have    
    
    specify filter_style='all' if all conditions should be met to be included in the return dictionary, specify filter_style='any' for including when any condition is met. Default is 'all'.
    """
    if type(filter_dict) != type(dict()):
        print("subset function error: type filter_dict should be dict")
        return
    return_dict = dictionary
    if filter_style=='any':
        a = {}
        for key, value in return_dict.items():            
            for i, j in filter_dict.items():
                if hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
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
    if filter_style == 'all':
        a = {key:value for key,value in return_dict.items()}         
        for key, value in return_dict.items():           
            for i, j in filter_dict.items():
                if hasattr(value, 'meta') & (type(value.meta) == type({})) & (i in value.meta.keys()):
                    try:
                        if value.__getattr__('meta')[i] not in j:
                            del a[key]
                    except:
                        pass
                elif hasattr(value, 'meta') & (type(value.meta) == type({})) & (i not in value.meta.keys()):
                    del a[key]
                elif hasattr(value,i)==False:
                    del a[key]
                else:
                    try:
                        if value.__getattr__(i) not in j:
                            del a[key]
                    except:
                        pass

    return a

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


def scan(directory, function, extension, target_dictionary):
    for j in os.listdir(directory):
        if j.split('.')[-1] == extension:
            processed = function(directory, j)
            target_dictionary[str(uuid.uuid4())] = processed
        elif (len(j.split('.'))) == 1:
            d = directory+j+'/'
            scan(d, function, extension, target_dictionary)
        else:
            pass
