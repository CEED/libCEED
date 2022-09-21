# -*- coding: UTF-8 -*-
from __future__ import with_statement

from six import u

from .asserts import verify_test_case
from junit_xml import TestCase as Case
from junit_xml import TestSuite as Suite
from junit_xml import decode
from .serializer import serialize_and_read


def test_init():
    ts, tcs = serialize_and_read(Suite("test", [Case("Test1")]))[0]
    verify_test_case(tcs[0], {"name": "Test1"})


def test_init_classname():
    ts, tcs = serialize_and_read(Suite("test", [Case(name="Test1", classname="some.class.name")]))[0]
    verify_test_case(tcs[0], {"name": "Test1", "classname": "some.class.name"})


def test_init_classname_time():
    ts, tcs = serialize_and_read(Suite("test", [Case(name="Test1", classname="some.class.name", elapsed_sec=123.345)]))[
        0
    ]
    verify_test_case(tcs[0], {"name": "Test1", "classname": "some.class.name", "time": ("%f" % 123.345)})


def test_init_classname_time_timestamp():
    ts, tcs = serialize_and_read(
        Suite("test", [Case(name="Test1", classname="some.class.name", elapsed_sec=123.345, timestamp=99999)])
    )[0]
    verify_test_case(
        tcs[0], {"name": "Test1", "classname": "some.class.name", "time": ("%f" % 123.345), "timestamp": ("%s" % 99999)}
    )


def test_init_stderr():
    ts, tcs = serialize_and_read(
        Suite("test", [Case(name="Test1", classname="some.class.name", elapsed_sec=123.345, stderr="I am stderr!")])
    )[0]
    verify_test_case(
        tcs[0], {"name": "Test1", "classname": "some.class.name", "time": ("%f" % 123.345)}, stderr="I am stderr!"
    )


def test_init_stdout_stderr():
    ts, tcs = serialize_and_read(
        Suite(
            "test",
            [
                Case(
                    name="Test1",
                    classname="some.class.name",
                    elapsed_sec=123.345,
                    stdout="I am stdout!",
                    stderr="I am stderr!",
                )
            ],
        )
    )[0]
    verify_test_case(
        tcs[0],
        {"name": "Test1", "classname": "some.class.name", "time": ("%f" % 123.345)},
        stdout="I am stdout!",
        stderr="I am stderr!",
    )


def test_init_disable():
    tc = Case("Disabled-Test")
    tc.is_enabled = False
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Disabled-Test"})


def test_init_failure_message():
    tc = Case("Failure-Message")
    tc.add_failure_info("failure message")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Failure-Message"}, failure_message="failure message")


def test_init_failure_output():
    tc = Case("Failure-Output")
    tc.add_failure_info(output="I failed!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Failure-Output"}, failure_output="I failed!")


def test_init_failure_type():
    tc = Case("Failure-Type")
    tc.add_failure_info(failure_type="com.example.Error")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Failure-Type"})

    tc.add_failure_info("failure message")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Failure-Type"}, failure_message="failure message", failure_type="com.example.Error"
    )


def test_init_failure():
    tc = Case("Failure-Message-and-Output")
    tc.add_failure_info("failure message", "I failed!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Failure-Message-and-Output"},
        failure_message="failure message",
        failure_output="I failed!",
        failure_type="failure",
    )


def test_init_error_message():
    tc = Case("Error-Message")
    tc.add_error_info("error message")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Error-Message"}, error_message="error message")


def test_init_error_output():
    tc = Case("Error-Output")
    tc.add_error_info(output="I errored!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Error-Output"}, error_output="I errored!")


def test_init_error_type():
    tc = Case("Error-Type")
    tc.add_error_info(error_type="com.example.Error")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Error-Type"})

    tc.add_error_info("error message")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Error-Type"}, error_message="error message", error_type="com.example.Error")


def test_init_error():
    tc = Case("Error-Message-and-Output")
    tc.add_error_info("error message", "I errored!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Error-Message-and-Output"},
        error_message="error message",
        error_output="I errored!",
        error_type="error",
    )


def test_init_skipped_message():
    tc = Case("Skipped-Message")
    tc.add_skipped_info("skipped message")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Skipped-Message"}, skipped_message="skipped message")


def test_init_skipped_output():
    tc = Case("Skipped-Output")
    tc.add_skipped_info(output="I skipped!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(tcs[0], {"name": "Skipped-Output"}, skipped_output="I skipped!")


def test_init_skipped_err_output():
    tc = Case("Skipped-Output")
    tc.add_skipped_info(output="I skipped!")
    tc.add_error_info(output="I skipped with an error!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Skipped-Output"}, skipped_output="I skipped!", error_output="I skipped with an error!"
    )


def test_init_skipped():
    tc = Case("Skipped-Message-and-Output")
    tc.add_skipped_info("skipped message", "I skipped!")
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Skipped-Message-and-Output"}, skipped_message="skipped message", skipped_output="I skipped!"
    )


def test_init_legal_unicode_char():
    tc = Case("Failure-Message")
    tc.add_failure_info(u("failure message with legal unicode char: [\x22]"))
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Failure-Message"}, failure_message=u("failure message with legal unicode char: [\x22]")
    )


def test_init_illegal_unicode_char():
    tc = Case("Failure-Message")
    tc.add_failure_info(u("failure message with illegal unicode char: [\x02]"))
    ts, tcs = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Failure-Message"}, failure_message=u("failure message with illegal unicode char: []")
    )


def test_init_utf8():
    tc = Case(
        name="Test äöü",
        classname="some.class.name.äöü",
        elapsed_sec=123.345,
        stdout="I am stdöüt!",
        stderr="I am stdärr!",
    )
    tc.add_skipped_info(message="Skipped äöü", output="I skippäd!")
    tc.add_error_info(message="Skipped error äöü", output="I skippäd with an error!")
    test_suite = Suite("Test UTF-8", [tc])
    ts, tcs = serialize_and_read(test_suite, encoding="utf-8")[0]
    verify_test_case(
        tcs[0],
        {
            "name": decode("Test äöü", "utf-8"),
            "classname": decode("some.class.name.äöü", "utf-8"),
            "time": ("%f" % 123.345),
        },
        stdout=decode("I am stdöüt!", "utf-8"),
        stderr=decode("I am stdärr!", "utf-8"),
        skipped_message=decode("Skipped äöü", "utf-8"),
        skipped_output=decode("I skippäd!", "utf-8"),
        error_message=decode("Skipped error äöü", "utf-8"),
        error_output=decode("I skippäd with an error!", "utf-8"),
    )


def test_init_unicode():
    tc = Case(
        name=decode("Test äöü", "utf-8"),
        classname=decode("some.class.name.äöü", "utf-8"),
        elapsed_sec=123.345,
        stdout=decode("I am stdöüt!", "utf-8"),
        stderr=decode("I am stdärr!", "utf-8"),
    )
    tc.add_skipped_info(message=decode("Skipped äöü", "utf-8"), output=decode("I skippäd!", "utf-8"))
    tc.add_error_info(message=decode("Skipped error äöü", "utf-8"), output=decode("I skippäd with an error!", "utf-8"))

    ts, tcs = serialize_and_read(Suite("Test Unicode", [tc]))[0]
    verify_test_case(
        tcs[0],
        {
            "name": decode("Test äöü", "utf-8"),
            "classname": decode("some.class.name.äöü", "utf-8"),
            "time": ("%f" % 123.345),
        },
        stdout=decode("I am stdöüt!", "utf-8"),
        stderr=decode("I am stdärr!", "utf-8"),
        skipped_message=decode("Skipped äöü", "utf-8"),
        skipped_output=decode("I skippäd!", "utf-8"),
        error_message=decode("Skipped error äöü", "utf-8"),
        error_output=decode("I skippäd with an error!", "utf-8"),
    )


def test_multiple_errors():
    """Tests multiple errors in one test case"""
    tc = Case("Multiple error", allow_multiple_subelements=True)
    tc.add_error_info("First error", "First error message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Multiple error"},
        errors=[{"message": "First error", "output": "First error message", "type": "error"}],
    )
    tc.add_error_info("Second error", "Second error message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Multiple error"},
        errors=[
            {"message": "First error", "output": "First error message", "type": "error"},
            {"message": "Second error", "output": "Second error message", "type": "error"},
        ],
    )


def test_multiple_failures():
    """Tests multiple failures in one test case"""
    tc = Case("Multiple failures", allow_multiple_subelements=True)
    tc.add_failure_info("First failure", "First failure message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Multiple failures"},
        failures=[{"message": "First failure", "output": "First failure message", "type": "failure"}],
    )
    tc.add_failure_info("Second failure", "Second failure message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Multiple failures"},
        failures=[
            {"message": "First failure", "output": "First failure message", "type": "failure"},
            {"message": "Second failure", "output": "Second failure message", "type": "failure"},
        ],
    )


def test_multiple_skipped():
    """Tests multiple skipped messages in one test case"""
    tc = Case("Multiple skipped", allow_multiple_subelements=True)
    tc.add_skipped_info("First skipped", "First skipped message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0], {"name": "Multiple skipped"}, skipped=[{"message": "First skipped", "output": "First skipped message"}]
    )
    tc.add_skipped_info("Second skipped", "Second skipped message")
    (_, tcs) = serialize_and_read(Suite("test", [tc]))[0]
    verify_test_case(
        tcs[0],
        {"name": "Multiple skipped"},
        skipped=[
            {"message": "First skipped", "output": "First skipped message"},
            {"message": "Second skipped", "output": "Second skipped message"},
        ],
    )
