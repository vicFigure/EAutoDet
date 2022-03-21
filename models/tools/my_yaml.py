import yaml

class PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

if __name__ == '__main__':
  doc = yaml.dump(tuple("foo bar baaz".split()))
  print(repr(doc))
  thing = yaml.load(doc, Loader=PrettySafeLoader)
  print(thing)

