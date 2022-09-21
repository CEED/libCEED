import codecs
import os
import tempfile
from xml.dom import minidom

from six import PY2

from junit_xml import to_xml_report_file, to_xml_report_string


def serialize_and_read(test_suites, to_file=False, prettyprint=False, encoding=None):
    """writes the test suite to an XML string and then re-reads it using minidom,
       returning => (test suite element, list of test case elements)"""
    try:
        iter(test_suites)
    except TypeError:
        test_suites = [test_suites]

    if to_file:
        fd, filename = tempfile.mkstemp(text=True)
        os.close(fd)
        with codecs.open(filename, mode="w", encoding=encoding) as f:
            to_xml_report_file(f, test_suites, prettyprint=prettyprint, encoding=encoding)
        print("Serialized XML to temp file [%s]" % filename)
        xmldoc = minidom.parse(filename)
        os.remove(filename)
    else:
        xml_string = to_xml_report_string(test_suites, prettyprint=prettyprint, encoding=encoding)
        if PY2:
            assert isinstance(xml_string, unicode)  # noqa: F821
        print("Serialized XML to string:\n%s" % xml_string)
        if encoding:
            xml_string = xml_string.encode(encoding)
        xmldoc = minidom.parseString(xml_string)

    def remove_blanks(node):
        for x in node.childNodes:
            if x.nodeType == minidom.Node.TEXT_NODE:
                if x.nodeValue:
                    x.nodeValue = x.nodeValue.strip()
            elif x.nodeType == minidom.Node.ELEMENT_NODE:
                remove_blanks(x)

    remove_blanks(xmldoc)
    xmldoc.normalize()

    ret = []
    suites = xmldoc.getElementsByTagName("testsuites")[0]
    for suite in suites.getElementsByTagName("testsuite"):
        cases = suite.getElementsByTagName("testcase")
        ret.append((suite, cases))
    return ret
