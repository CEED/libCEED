def verify_test_case(  # noqa: E302
    test_case_element,
    expected_attributes,
    error_message=None,
    error_output=None,
    error_type=None,
    failure_message=None,
    failure_output=None,
    failure_type=None,
    skipped_message=None,
    skipped_output=None,
    stdout=None,
    stderr=None,
    errors=None,
    failures=None,
    skipped=None,
):
    for k, v in expected_attributes.items():
        assert test_case_element.attributes[k].value == v

    for k in test_case_element.attributes.keys():
        assert k in expected_attributes.keys()

        if stderr:
            assert test_case_element.getElementsByTagName("system-err")[0].firstChild.nodeValue.strip() == stderr
        if stdout:
            assert test_case_element.getElementsByTagName("system-out")[0].firstChild.nodeValue.strip() == stdout

        _errors = test_case_element.getElementsByTagName("error")
        if error_message or error_output:
            assert len(_errors) > 0
        elif errors:
            assert len(errors) == len(_errors)
        else:
            assert len(_errors) == 0

        if error_message:
            assert _errors[0].attributes["message"].value == error_message

        if error_type and _errors:
            assert _errors[0].attributes["type"].value == error_type

        if error_output:
            assert _errors[0].firstChild.nodeValue.strip() == error_output

        for error_exp, error_r in zip(errors or [], _errors):
            assert error_r.attributes["message"].value == error_exp["message"]
            assert error_r.firstChild.nodeValue.strip() == error_exp["output"]
            assert error_r.attributes["type"].value == error_exp["type"]

        _failures = test_case_element.getElementsByTagName("failure")
        if failure_message or failure_output:
            assert len(_failures) > 0
        elif failures:
            assert len(failures) == len(_failures)
        else:
            assert len(_failures) == 0

        if failure_message:
            assert _failures[0].attributes["message"].value == failure_message

        if failure_type and _failures:
            assert _failures[0].attributes["type"].value == failure_type

        if failure_output:
            assert _failures[0].firstChild.nodeValue.strip() == failure_output

        for failure_exp, failure_r in zip(failures or [], _failures):
            assert failure_r.attributes["message"].value == failure_exp["message"]
            assert failure_r.firstChild.nodeValue.strip() == failure_exp["output"]
            assert failure_r.attributes["type"].value == failure_exp["type"]

        _skipped = test_case_element.getElementsByTagName("skipped")
        if skipped_message or skipped_output:
            assert len(_skipped) > 0
        elif skipped:
            assert len(skipped) == len(_skipped)
        else:
            assert len(_skipped) == 0

        for skipped_exp, skipped_r in zip(skipped or [], _skipped):
            assert skipped_r.attributes["message"].value == skipped_exp["message"]
            assert skipped_r.firstChild.nodeValue.strip() == skipped_exp["output"]
