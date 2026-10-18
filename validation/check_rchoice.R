# =====================================================================================================================
# Checks 1 and 2 of Appendix E with R: fixed-parameter logit (glm) and random parameters binary logit (Rchoice).
# Rchoice is given the same standard Halton draws as the Python estimator: prime 2 for new_year, prime 3 for
# Northeastern, first 10 points dropped, crash i uses points i*R .. i*R+R-1.
#
# Run after   python validation/check_data.py   and   python validation/estimate_python.py 1 2
# usage (from the repository root):   Rscript validation/check_rchoice.R
# needs R packages Rchoice, maxLik, sandwich, jsonlite
# writes validation/output/rchoice_check1.json, rchoice_check2.json, rchoice_check2.csv
# =====================================================================================================================
suppressPackageStartupMessages({ library(Rchoice); library(maxLik); library(sandwich); library(jsonlite) })
out <- file.path("validation", "output")
dat <- read.csv(file.path(out, "S1_check_data.csv"))
spec <- fromJSON(file.path("validation", "check_specs.json"))
xfix <- spec$check2$fixed; xrnd <- spec$check2$random
fml <- as.formula(paste("sev ~ 0 +", paste(c(xfix, xrnd), collapse = " + ")))

# ---- check 1: fixed-parameter logit
py1 <- fromJSON(file.path(out, "python_check1.json"))
fix <- glm(fml, data = dat, family = binomial("logit"))
r1 <- list(check = 1, LL_python = py1$LL, LL_glm = as.numeric(logLik(fix)), difference = as.numeric(logLik(fix)) - py1$LL)
write_json(r1, file.path(out, "rchoice_check1.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat(sprintf("check 1: LL glm = %.4f, Python = %.4f\n", r1$LL_glm, py1$LL))

# ---- check 2: random parameters binary logit, 500 Halton draws
py <- read.csv(file.path(out, "python_check2.csv")); py2 <- fromJSON(file.path(out, "python_check2.json"))
bpy <- py$estimate[match(c(xfix, xrnd, paste0("SD:", xrnd)), py$parameter)]
names(bpy) <- c(xfix, xrnd, paste0("sd.", xrnd))
common <- list(formula = fml, data = dat, family = binomial("logit"), ranp = setNames(rep("n", length(xrnd)), xrnd),
               R = spec$check2$draws, haltons = list(prime = c(2, 3), drop = c(10, 10)))
m0 <- do.call(Rchoice, c(common, list(start = bpy, iterlim = 0, method = "bfgs")))       # at the Python estimates
g0 <- max(abs(colSums(estfun(m0))))
t0 <- Sys.time()
m1 <- do.call(Rchoice, c(common, list(method = "bfgs")))                                 # own estimation, SD start 0.1
b1 <- coef(m1); b1[grepl("^sd\\.", names(b1))] <- abs(b1[grepl("^sd\\.", names(b1))])
se1 <- sqrt(diag(vcov(m1)))
cmp <- data.frame(parameter = c(xfix, xrnd, paste0("SD:", xrnd)), python_estimate = unname(bpy[names(b1)]),
                  python_se = py$se[match(c(xfix, xrnd, paste0("SD:", xrnd)), py$parameter)],
                  rchoice_estimate = unname(b1), rchoice_se = unname(se1),
                  rchoice_se_opg = sqrt(diag(solve(crossprod(estfun(m1))))), rchoice_se_sandwich = sqrt(diag(sandwich(m1))))
write.csv(cmp, file.path(out, "rchoice_check2.csv"), row.names = FALSE)
r2 <- list(check = 2, rchoice_version = as.character(packageVersion("Rchoice")), LL_python = py2$LL,
           LL_rchoice_at_python_estimates = as.numeric(logLik(m0)), difference = as.numeric(logLik(m0)) - py2$LL,
           max_abs_rchoice_gradient_at_python_estimates = g0, LL_rchoice_estimated = as.numeric(logLik(m1)),
           max_abs_coef_diff = max(abs(cmp$rchoice_estimate - cmp$python_estimate)),
           max_abs_se_diff = max(abs(cmp$rchoice_se - cmp$python_se)),
           estimation_minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
write_json(r2, file.path(out, "rchoice_check2.json"), auto_unbox = TRUE, digits = NA, pretty = TRUE)
print(r2)
