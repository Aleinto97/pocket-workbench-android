// GenieX's Android plugin can crash while unloading at geniex_deinit(), after
// the benchmark has already closed its JSON report. Each device is measured in
// its own process, so no in-process teardown is needed after the report.
#include <stdio.h>
#include <stdlib.h>

int geniex_deinit(void) {
    fflush(NULL);
    _Exit(0);
}
