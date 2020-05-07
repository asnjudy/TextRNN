from functools import reduce

make_json = lambda blanks, obj: \
    (lambda t, cut: \
         'null' if obj == None \
             else str(obj) if t in (int, float) \
             else ('true' if obj else 'false') if t == bool \
             else '"%s"' % obj if t == str \
             else '[' + cut(reduce(lambda r, x: r + ',\n' + ' ' * (blanks + 2) + make_json(blanks + 2, x), obj, '')) \
                  + '\n' + ' ' * blanks + ']' if t in (list, tuple) \
             else '{' + cut(
             reduce(lambda r, x: r + ',\n' + ' ' * (blanks + 2) + '"%s" : ' % x + make_json(blanks + 2, obj[x]), \
                    sorted(filter(lambda x: type(x) == str, obj.keys())), '')) + '\n' + ' ' * blanks + '}' if t == dict \
             else reduce(lambda r, x: r + '%02x' % x, list(map(int, obj)), '"') + '"' if t == bytes \
             else '{' + cut(reduce(lambda r, x: \
                                       r + ',\n' + ' ' * (blanks + 2) + '"%s" : ' % x + make_json(blanks + 2,
                                                                                                  obj.__dict__[x]), \
                                   sorted(filter(lambda x: len(x) < 4 or x[:2] != '__' \
                                                           or x[-2:] != '__', obj.__dict__.keys())),
                                   '')) + '\n' + ' ' * blanks + '}') \
        (type(obj), lambda x: x if x == '' else x[1:])
print_json = lambda obj, fprint: fprint(make_json(0, obj))
