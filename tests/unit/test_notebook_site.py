from tsfmxai.visualization.notebook_site import notebook_navigation, prepare_notebook_site


def test_directory_tree_preserves_names_and_excludes_checkpoints(tmp_path):
    paths = ["experiments/group_name/AR/02_model.ipynb", "experiments/group_name/01_model.ipynb",
             "experiments/.ipynb_checkpoints/hidden.ipynb", "experiments_local/local.ipynb"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original notebook bytes")
    tree = notebook_navigation(tmp_path)
    group = tree[0]["contents"][0]
    assert group["section"] == "group_name"
    assert group["contents"][1]["section"] == "AR"
    assert group["contents"][1]["contents"][0]["text"] == "02_model.ipynb"
    prepare_notebook_site(tmp_path)
    html = (tmp_path / "outputs/notebook-tree.html").read_text()
    assert 'experiments/group_name/AR/02_model.html' in html
    assert 'hidden' not in html and 'local.ipynb' not in html
    for name in paths:
        assert (tmp_path / name).read_bytes() == b"original notebook bytes"


def test_empty_folders_and_html_escaping(tmp_path):
    (tmp_path / "experiments/empty").mkdir(parents=True)
    folder = tmp_path / "experiments/a & b"
    folder.mkdir()
    (folder / "a & b.ipynb").write_text("{}")
    prepare_notebook_site(tmp_path)
    html = (tmp_path / "outputs/notebook-tree.html").read_text()
    assert 'empty' not in html
    assert 'a &amp; b' in html
    assert 'a%20%26%20b.html' in html
