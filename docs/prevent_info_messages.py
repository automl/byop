"""The module is a hook which disables warnings and log messages which pollute the
doc build output.

One possible downside is if one of these modules ends up giving an actual
error, such as OpenML failing to retrieve a dataset. I tried to make sure ERROR
log message are still allowed through.
"""
import logging
import warnings
from typing import Any

import mkdocs
import mkdocs.plugins
import mkdocs.structure.pages

log = logging.getLogger("mkdocs")


@mkdocs.plugins.event_priority(-50)
def on_startup(**kwargs: Any):
    # We get a load of deprecation warnings from SMAC
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # ConvergenceWarning from sklearn
    warnings.filterwarnings("ignore", module="sklearn")

    # There's also one code cell in `scheduling.md` that
    # demonstrates that the scheduler needs to be running to submit a task.
    # This casuses a `log.error` to be emitted, which we don't want.


def on_pre_page(
    page: mkdocs.structure.pages.Page,
    config: Any,
    files: Any,
) -> mkdocs.structure.pages.Page | None:
    # NOTE: mkdocs says they're always normalized to be '/' seperated
    # which means this should work on windows as well.

    # This error is actually demonstrated to the user which causes amltk
    # to log the error. I don't know how to disable it for that one code cell
    # put I can at least limit it to the file in which it's in.
    if page.file.src_uri == "guides/scheduling.md":
        scheduling_logger = logging.getLogger("amltk.scheduling.task")
        scheduling_logger.setLevel(logging.CRITICAL)

    logging.getLogger("smac").setLevel(logging.ERROR)
    logging.getLogger("openml").setLevel(logging.ERROR)
    return page
