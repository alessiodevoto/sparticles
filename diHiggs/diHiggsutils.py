def make_tuple(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        return (x,)


#defining function for converting h5 file to have all object features in the correct format.
def convert(x):
     if isinstance(x, dict): #check if x is a dictionary
         if len(x) == 1: 
             item = list(x.values())[0] #extract the value of the single key of the dict
             if isinstance(item, list): #if the value is a list return the first element
                 return item[0]
             else:
                 return item 
     else:
         return x #if x is not a dictionary return itself
