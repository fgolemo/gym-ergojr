import os
import xml
from xml.dom.minidom import parse

from gym_ergojr.utils.xacro_standalone import process_includes, eval_self_contained


class URDF(object):
    def __init__(self, path, force_recompile=False):
        self.path = ""

        # path should not contain file ending, like .xml, .urdf, .xacro or whatever

        if ".xacro" in path:
            self.path = self.convertToURDF(path)
        elif ".urdf" in path:
            self.path = path
        else:
            if os.path.isfile(path + ".urdf") and not force_recompile:
                self.path = path + ".urdf"
            else:
                self.path = self.convertToURDF(path + ".xacro")

    def get_path(self):
        return str(self)

    def __str__(self):
        return self.path

    def convertToURDF(self, path):
        if not os.path.isfile(path):
            path = path + ".xml"

        if not os.path.isfile(path):
            raise FileNotFoundError(
                "I was looking for '{}' (and also without the .xml) "
                "but couldn't find it".format(path)
            )

        f = open(path)
        doc = None
        try:
            doc = parse(f)

        except xml.parsers.expat.ExpatError:
            print("xml parsing error")
            raise
        finally:
            f.close()

        process_includes(doc, os.path.dirname(path))
        eval_self_contained(doc)

        if not ".urdf" in path:
            output_path = path.replace(".xacro", ".urdf")
        else:
            output_path = path.replace(".xacro", "")

        with open(output_path, "w") as output_file:
            print("Debug: Converting Xacro file - don't do this in production")
            output_file.write(doc.toprettyxml(indent='  '))

        return output_path
