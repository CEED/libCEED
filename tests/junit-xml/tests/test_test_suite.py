# -*- coding: UTF-8 -*-
from __future__ import with_statement

import textwrap
import warnings

import pytest
from six import PY2, StringIO

from .asserts import verify_test_case
from junit_xml import TestCase as Case
from junit_xml import TestSuite as Suite
from junit_xml import decode, to_xml_report_string
from .serializer import serialize_and_read


def test_single_suite_single_test_case():
    with pytest.raises(TypeError) as excinfo:
        serialize_and_read(Suite("test", Case("Test1")), to_file=True)[0]
    assert str(excinfo.value) == "test_cases must be a list of test cases"


def test_single_suite_no_test_cases():
    properties = {"foo": "bar"}
    package = "mypackage"
    timestamp = 1398382805

    ts, tcs = serialize_and_read(
        Suite(
            name="test",
            test_cases=[],
            hostname="localhost",
            id=1,
            properties=properties,
            package=package,
            timestamp=timestamp,
        ),
        to_file=True,
        prettyprint=True,
    )[0]
    assert ts.tagName == "testsuite"
    assert ts.attributes["package"].value == package
    assert ts.attributes["timestamp"].value == str(timestamp)
    assert ts.childNodes[0].childNodes[0].attributes["name"].value == "foo"
    assert ts.childNodes[0].childNodes[0].attributes["value"].value == "bar"


def test_single_suite_no_test_cases_utf8():
    properties = {"foö": "bär"}
    package = "mypäckage"
    timestamp = 1398382805

    test_suite = Suite(
        name="äöü",
        test_cases=[],
        hostname="löcalhost",
        id="äöü",
        properties=properties,
        package=package,
        timestamp=timestamp,
    )
    ts, tcs = serialize_and_read(test_suite, to_file=True, prettyprint=True, encoding="utf-8")[0]
    assert ts.tagName == "testsuite"
    assert ts.attributes["package"].value == decode(package, "utf-8")
    assert ts.attributes["timestamp"].value == str(timestamp)
    assert ts.childNodes[0].childNodes[0].attributes["name"].value == decode("foö", "utf-8")
    assert ts.childNodes[0].childNodes[0].attributes["value"].value == decode("bär", "utf-8")


def test_single_suite_no_test_cases_unicode():
    properties = {decode("foö", "utf-8"): decode("bär", "utf-8")}
    package = decode("mypäckage", "utf-8")
    timestamp = 1398382805

    ts, tcs = serialize_and_read(
        Suite(
            name=decode("äöü", "utf-8"),
            test_cases=[],
            hostname=decode("löcalhost", "utf-8"),
            id=decode("äöü", "utf-8"),
            properties=properties,
            package=package,
            timestamp=timestamp,
        ),
        to_file=True,
        prettyprint=True,
        encoding="utf-8",
    )[0]
    assert ts.tagName == "testsuite"
    assert ts.attributes["package"].value == package
    assert ts.attributes["timestamp"].value, str(timestamp)
    assert ts.childNodes[0].childNodes[0].attributes["name"].value == decode("foö", "utf-8")
    assert ts.childNodes[0].childNodes[0].attributes["value"].value == decode("bär", "utf-8")


def test_single_suite_to_file():
    ts, tcs = serialize_and_read(Suite("test", [Case("Test1")]), to_file=True)[0]
    verify_test_case(tcs[0], {"name": "Test1"})


def test_single_suite_to_file_prettyprint():
    ts, tcs = serialize_and_read(Suite("test", [Case("Test1")]), to_file=True, prettyprint=True)[0]
    verify_test_case(tcs[0], {"name": "Test1"})


def test_single_suite_prettyprint():
    ts, tcs = serialize_and_read(Suite("test", [Case("Test1")]), to_file=False, prettyprint=True)[0]
    verify_test_case(tcs[0], {"name": "Test1"})


def test_single_suite_to_file_no_prettyprint():
    ts, tcs = serialize_and_read(Suite("test", [Case("Test1")]), to_file=True, prettyprint=False)[0]
    verify_test_case(tcs[0], {"name": "Test1"})


def test_multiple_suites_to_file():
    tss = [Suite("suite1", [Case("Test1")]), Suite("suite2", [Case("Test2")])]
    suites = serialize_and_read(tss, to_file=True)

    assert suites[0][0].attributes["name"].value == "suite1"
    verify_test_case(suites[0][1][0], {"name": "Test1"})

    assert suites[1][0].attributes["name"].value == "suite2"
    verify_test_case(suites[1][1][0], {"name": "Test2"})


def test_multiple_suites_to_string():
    tss = [Suite("suite1", [Case("Test1")]), Suite("suite2", [Case("Test2")])]
    suites = serialize_and_read(tss)

    assert suites[0][0].attributes["name"].value == "suite1"
    verify_test_case(suites[0][1][0], {"name": "Test1"})

    assert suites[1][0].attributes["name"].value == "suite2"
    verify_test_case(suites[1][1][0], {"name": "Test2"})


def test_attribute_time():
    tss = [
        Suite(
            "suite1",
            [
                Case(name="Test1", classname="some.class.name", elapsed_sec=123.345),
                Case(name="Test2", classname="some2.class.name", elapsed_sec=123.345),
            ],
        ),
        Suite("suite2", [Case("Test2")]),
    ]
    suites = serialize_and_read(tss)

    assert suites[0][0].attributes["name"].value == "suite1"
    assert suites[0][0].attributes["time"].value == "246.69"

    assert suites[1][0].attributes["name"].value == "suite2"
    # here the time in testsuite is "0" even there is no attribute time for
    # testcase
    assert suites[1][0].attributes["time"].value == "0"


def test_attribute_disable():
    tc = Case("Disabled-Test")
    tc.is_enabled = False
    tss = [Suite("suite1", [tc])]
    suites = serialize_and_read(tss)

    assert suites[0][0].attributes["disabled"].value == "1"


def test_stderr():
    suites = serialize_and_read(Suite(name="test", stderr="I am stderr!", test_cases=[Case(name="Test1")]))[0]
    assert suites[0].getElementsByTagName("system-err")[0].firstChild.data == "I am stderr!"


def test_stdout_stderr():
    suites = serialize_and_read(
        Suite(name="test", stdout="I am stdout!", stderr="I am stderr!", test_cases=[Case(name="Test1")])
    )[0]
    assert suites[0].getElementsByTagName("system-err")[0].firstChild.data == "I am stderr!"
    assert suites[0].getElementsByTagName("system-out")[0].firstChild.data == "I am stdout!"


def test_no_assertions():
    suites = serialize_and_read(Suite(name="test", test_cases=[Case(name="Test1")]))[0]
    assert not suites[0].getElementsByTagName("testcase")[0].hasAttribute("assertions")


def test_assertions():
    suites = serialize_and_read(Suite(name="test", test_cases=[Case(name="Test1", assertions=5)]))[0]
    assert suites[0].getElementsByTagName("testcase")[0].attributes["assertions"].value == "5"

    # @todo: add more tests for the other attributes and properties


def test_to_xml_string():
    test_suites = [
        Suite(name="suite1", test_cases=[Case(name="Test1")]),
        Suite(name="suite2", test_cases=[Case(name="Test2")]),
    ]
    xml_string = to_xml_report_string(test_suites)
    if PY2:
        assert isinstance(xml_string, unicode)  # noqa: F821
    expected_xml_string = textwrap.dedent(
        """
        <?xml version="1.0" ?>
        <testsuites disabled="0" errors="0" failures="0" tests="2" time="0.0">
        \t<testsuite disabled="0" errors="0" failures="0" name="suite1" skipped="0" tests="1" time="0">
        \t\t<testcase name="Test1"/>
        \t</testsuite>
        \t<testsuite disabled="0" errors="0" failures="0" name="suite2" skipped="0" tests="1" time="0">
        \t\t<testcase name="Test2"/>
        \t</testsuite>
        </testsuites>
    """.strip(
            "\n"
        )
    )
    assert xml_string == expected_xml_string


def test_to_xml_string_test_suites_not_a_list():
    test_suites = Suite("suite1", [Case("Test1")])

    with pytest.raises(TypeError) as excinfo:
        to_xml_report_string(test_suites)
    assert str(excinfo.value) == "test_suites must be a list of test suites"


def test_deprecated_to_xml_string():
    with warnings.catch_warnings(record=True) as w:
        Suite.to_xml_string([])
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Testsuite.to_xml_string is deprecated" in str(w[0].message)


def test_deprecated_to_file():
    with warnings.catch_warnings(record=True) as w:
        Suite.to_file(StringIO(), [])
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "Testsuite.to_file is deprecated" in str(w[0].message)
