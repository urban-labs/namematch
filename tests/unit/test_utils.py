from namematch.utils.utils import clean_nn_string, build_blockstring


def test_create_nm_record_id():
    pass


def test_clean_nn_string():

    # test JR and .
    assert clean_nn_string("JOHN SMITH JR.") == 'JOHN SMITH'

     # test STRIP, III, and -
    assert clean_nn_string(" JOHN  SM-ITH III") == 'JOHN SMITH'

    # test strip non A-Z (uppercase)
    assert clean_nn_string("JOHN Smith") == 'JOHN S'

    # test that uppercase isn't handled
    assert clean_nn_string("JOHN smith") != 'JOHN SMITH'


def test_build_blockstring(an_df):
    scheme = {
        "cosine_distance": {
            "variables": ["first_name", "last_name"],
	    "relative_weight_of_second_variable": 1.2
        },
        "edit_distance": {"variable": "dob"}
    }

    # get blockstrings
    blockstring_col = build_blockstring(an_df, scheme)

    # test
    assert an_df.shape[0] == len(blockstring_col)
    expected_num_cols = len(scheme['cosine_distance']['variables'] + [scheme['edit_distance']['variable']])
    num_cols = len(blockstring_col[0].split("::"))
    assert expected_num_cols == num_cols


def test_determine_model_to_use():
    pass


def test_get_nn_string_from_blockstring():
    pass


def test_get_ed_string_from_blockstring():
    pass


def test_get_endpoints():
    pass


