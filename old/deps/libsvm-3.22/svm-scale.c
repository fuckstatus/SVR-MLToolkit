#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

void exit_with_help()
{
	printf(
	"Usage: svm-scale [options] data_filename\n"
	"options:\n"
	"-l lower : x scaling lower limit (default -1)\n"
	"-u upper : x scaling upper limit (default +1)\n"
	"-y y_lower y_upper : y scaling limits (default: no y scaling)\n"
	"-s save_filename : save scaling parameters to save_filename\n"
	"-r restore_filename : restore scaling parameters from restore_filename\n"
	);
	exit(1);
}

char *line = NULL;
int max_lin