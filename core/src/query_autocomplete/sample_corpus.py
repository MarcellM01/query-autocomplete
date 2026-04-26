"""Bundled sample corpus for quick try-out without supplying your own data.

Covers a software-development / code-writing domain with fewer, longer
documents so the sample works well for in-memory, artifact, and adaptive
mutable-store demos.
"""

from query_autocomplete.models import Document


SAMPLE_TRAINING_DOCS: list[Document] = [
    Document(
        text=(
            "open a new file in the editor and check the contents before you start editing\n"
            "save the file before closing the editor so you do not lose any recent changes\n"
            "split the editor into two panels when you want to compare the implementation and the test side by side\n"
            "search and replace across all files after you rename a symbol and need to update the surrounding call sites\n"
            "find all references to this function and go to the definition of the selected symbol before changing shared logic"
        ),
        doc_id="sample-editor-workflow",
    ),
    Document(
        text=(
            "create a new function to handle the request and write a type annotation for each parameter before wiring it into the service\n"
            "create a new class for the data model when the response shape becomes hard to reason about and refactor the class to separate concerns\n"
            "extract the logic into a separate helper function when one block grows too large and split the large function into smaller functions with descriptive names\n"
            "add a parameter to the function signature only when the caller truly needs control and add a return type annotation to make the interface easier to read\n"
            "rename the variable to a more descriptive name when the intent is unclear and review the surrounding code for other names that should become more explicit"
        ),
        doc_id="sample-code-workflow",
    ),
    Document(
        text=(
            "run the test suite and check for failures before you push the branch to the remote repository\n"
            "run the failing test in isolation when the full suite is noisy and add a test case for the edge case that caused the regression\n"
            "write a test to verify the error handling path and mock the external dependency in the test when the integration is slow or flaky\n"
            "check the test coverage for this module after a refactor so you can confirm the important branches are still exercised\n"
            "commit the changes with a descriptive message only after the tests pass and the diff is small enough to review comfortably"
        ),
        doc_id="sample-testing-workflow",
    ),
    Document(
        text=(
            "create a new branch for the feature before you start coding and push the branch when the first clean milestone is ready for review\n"
            "review the diff before committing and stash the local changes before pulling when the branch has drifted from main\n"
            "push the branch and open a pull request after you have written the tests and updated the documentation for the new behavior\n"
            "merge the feature branch into main only after the review comments are addressed and the automated checks are green\n"
            "revert the last commit to fix the mistake when a change lands incorrectly and you need a fast, explicit recovery path"
        ),
        doc_id="sample-git-workflow",
    ),
    Document(
        text=(
            "add a docstring to the public function so the intent is clear when another engineer reads the module later\n"
            "add an example to the documentation and write a usage example for the new feature when the API shape is not obvious from the function name alone\n"
            "update the readme with the new instructions after the workflow changes and document the parameters and return values for every public entrypoint\n"
            "update the changelog for the new release when the behavior changes in a user-visible way and include a short migration note when an older pattern is no longer recommended\n"
            "document the tradeoffs behind the implementation when a simpler-looking alternative was rejected so future edits do not accidentally remove an important constraint"
        ),
        doc_id="sample-documentation-workflow",
    ),
]
