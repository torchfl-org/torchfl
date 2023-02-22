.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/vivekkhimani/torchfl/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features.
Anything tagged with "enhancement", "help wanted",
and "feature" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

torchfl could always use more documentation, whether as part of the
official torchfl docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/vivekkhimani/torchfl/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `torchfl` for local development.

1. Fork the `torchfl` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:<your_username_here>/torchfl.git

3. Install Poetry to manage dependencies and virtual environments from https://python-poetry.org/docs/.
4. Install the project dependencies using::

    $ poetry install

5. To add a new dependency to the project, use::

    $ poetry add <dependency_name>

6. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally and maintain them on your own branch.

7. When you're done making changes, check that your changes pass the tests::

    $ pytest tests

   If you want to run a specific test file, use::

    $ pytest <path-to-the-file>

   If your changes are not covered by the tests, please add tests.

8. The pre-commit hooks will be run before every commit.
   If you want to run them manually, use::

    $ pre-commit run --all

9. Commit your changes and push your branch to GitHub::

    $ git add --all
    $ git commit -m "Your detailed description of your changes."
    $ git push origin <name-of-your-bugfix-or-feature>

10. Submit a pull request through the GitHub website.
11. Once the pull request has been submitted,
    the CI pipelines will be triggered on GitHub Actions,
    All of them must pass before one of the maintainers
    can review the request and perform the merge.

Pull Request Guidelines
----------------------------

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.

3. The pull request should work for Python3, and for PyPy. Check
   https://travis-ci.com/vivekkhimani/torchfl/pull_requests
   and make sure that the tests pass for all supported Python versions.
