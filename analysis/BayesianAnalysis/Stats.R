## Title: Informativeness ratings of human vs. LLM referring expressions
## Author: Kristina Kobrock
## Description: Analyze informativeness for INLG submission
## Last Edit: 2026-09-03

## Ordinal regression analysis following tutorials by Bürkner & Vuorre (2019) 
# https://doi.org/10.1177/2515245918823199
# and Nicenboim (2026):
# https://bruno.nicenboim.me/posts/posts/2026-01-09-ordinal-models/index.html

# loading libraries-----------------------------------------------------
library(tidyverse)
library(tidybayes)
library(bayestestR)
library(jsonlite)
library(brms)
library(lme4)
library(posterior)
#devtools::install_github("michael-franke/aida-package")
library(aida) # theme for plotting
theme_set(theme_aida())
          
set.seed(116)

## getting the path of your current open file
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

# loading and preparing the data for modeling--------------------------
df <- read_json('../../output/intermediate_results/human_eval_per_item.json', simplifyVector = TRUE)

df$condition <- factor(df$condition, levels = c("SPLIT", "FAR", "CLOSE"))

df <- df %>% mutate(human_vs_model = ifelse(architecture == "human", "human", "model"))
df$human_vs_model <- factor(df$human_vs_model, levels = c("human", "model"))

# transform scale from -2, -1, 0, 1, 2 back to 1, 2, 3, 4, 5
df$informativeness_likert <- df$description_informativeness + 3

# excluding 4B models for better estimates (4B was shown to be unreliable in previous analyses)
df_exc4b <- df %>% filter(system != "Qwen3.5-4B_thinking" & system != "gemma-4-E4B-it_thinking"
                          & system != "gemma-4-E4B-it" & system != "Qwen3.5-4B")

# model----------------------------------------------------------------
fit_informativeness_exc <- brm(informativeness_likert ~ human_vs_model * condition,
                               data = df_exc4b,
                               family = cumulative(link = "probit"),
                               iter = 4000,
                               warmup = 1000,
                               cores = getOption("mc.cores", 4),
                               file = "fit_informativeness_exc")
summary(fit_informativeness_exc)
pp_check(fit_informativeness_exc)
pp_check(fit_informativeness_exc, "bars_grouped", group="condition")
pp_check(fit_informativeness_exc, "bars_grouped", group="human_vs_model")
describe_posterior(fit_informativeness_exc)

# visualization of logistic distribution with thresholds---------
thres <- c(fixef(fit_informativeness_exc)[1], fixef(fit_informativeness_exc)[2], 
           fixef(fit_informativeness_exc)[3], fixef(fit_informativeness_exc)[4])

category <- factor(1:5, labels = c("1", "2", "3", "4", "5"))

x_vals <- seq(-6, 6, length.out = 500)
norm_data <- data.frame(
  x = x_vals,
  y = dlogis(x_vals)
)

category_positions <- c(-Inf, thres, Inf)
category_midpoints <- (category_positions[-1] + category_positions[-length(category_positions)]) / 2
category_midpoints[1] <- max(-6, category_midpoints[1])
category_midpoints[5] <- min(6, category_midpoints[5])

threshold_labels <- data.frame(
  x = thres,
  y = dlogis(thres) + 0.02,
  label = paste0("τ[", 1:4, "]")
)

category_labels <- data.frame(
  x = category_midpoints,
  y = 0.02,
  label = as.character(category)
)

p1 <- ggplot(norm_data, aes(x = x, y = y)) +
  geom_line(linewidth = 1, color = "steelblue") +
  geom_vline(xintercept = thres, linetype = "dashed", color = "red", linewidth = 0.8) +
  geom_text(data = threshold_labels, aes(x = x, y = y, label = label),
            parse = TRUE, size = 4.5, color = "red", hjust = -0.2, vjust = 0) +
  geom_text(data = category_labels, aes(x = x, y = y, label = label),
            size = 3.5, color = "black", hjust = 0.5) +
  labs(title = "Standard Logistic Distribution with Thresholds",
       x = "Latent Variable",
       y = "Density") +
  theme_minimal(base_size = 12)

print(p1)
----------
# extracting fitted values ------------------------------------------
new_data <- data.frame(human_vs_model = rep(c("human", "model"), each=3),
                       condition = rep(c("SPLIT", "FAR", "CLOSE"), times=2))
# extract fitted values
fitted_probs <- fitted(fit_informativeness_exc, newdata = new_data)
# sidenote: instead of extracting the fitted values from the model, you can also
# use a posterior predictive distribution here using posterior_epred (code would
# need to be adapted to calculate the means and quartiles)

fitted_df <- data.frame(
  human_vs_model = rep(c("human", "model"), each=15),
  condition = rep(c("SPLIT", "FAR", "CLOSE"), each=5),
  category = rep(1:5, times = 6),
  probability = c(fitted_probs[1, "Estimate", ], fitted_probs[2, "Estimate",],
                  fitted_probs[3, "Estimate", ], fitted_probs[4, "Estimate", ],
                  fitted_probs[5, "Estimate", ], fitted_probs[6, "Estimate", ]),
  lower = c(fitted_probs[1, "Q2.5",], fitted_probs[2 , "Q2.5",],
            fitted_probs[3 , "Q2.5",], fitted_probs[4 , "Q2.5",],
            fitted_probs[5 , "Q2.5",], fitted_probs[6 , "Q2.5",]),
  upper = c(fitted_probs[1, "Q97.5",], fitted_probs[2 , "Q97.5",],
            fitted_probs[3 , "Q97.5",], fitted_probs[4 , "Q97.5",],
            fitted_probs[5 , "Q97.5",], fitted_probs[6 , "Q97.5",])
)

# statistical analysis of relevant differences -----------------------
new_data <- data.frame(human_vs_model = rep(c("human", "model"), each=3),
                       condition = rep(c("SPLIT", "FAR", "CLOSE"), times=2)
                       )
# posterior predictive distribution
posterior_probs <- posterior_epred(fit_informativeness_exc, newdata = new_data)
posterior_probs[,3,1] # (draw, cond_x_model, category)

# far, optimal (3), human vs. model
describe_posterior(posterior_probs[,2,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,5,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,2,3] - posterior_probs[,5,3], rope_range = rope_range(fit_informativeness_exc))

# far, underinformative (1), human vs. model
describe_posterior(posterior_probs[,2,1], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,5,1], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,2,1] - posterior_probs[,5,1], rope_range = rope_range(fit_informativeness_exc))

# far, overinformative (5), human vs. model
describe_posterior(posterior_probs[,2,5], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,5,5], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,2,5] - posterior_probs[,5,5], rope_range = rope_range(fit_informativeness_exc))

# split, optimal (3), human vs. model
describe_posterior(posterior_probs[,1,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,4,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,1,3] - posterior_probs[,4,3], rope_range = rope_range(fit_informativeness_exc))

# close, optimal (3), human vs. model
describe_posterior(posterior_probs[,3,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,6,3], rope_range = rope_range(fit_informativeness_exc))
describe_posterior(posterior_probs[,3,3] - posterior_probs[,6,3], rope_range = rope_range(fit_informativeness_exc))

# plotting informativeness ratings ------------------------------------
# transform ratings back to (-2, -1, 0, 1, 2)
fitted_df$category <- as.character(as.numeric(fitted_df$category) -3)

fitted_df %>% 
  mutate(condition = ifelse(condition == "FAR", "far", 
                            ifelse(condition == "SPLIT", "split", 
                                   ifelse(condition == "CLOSE", "close", NA)))) %>% 
  mutate(condition = factor(condition, levels = c("far", "split", "close"))) %>% 
  mutate(category = factor(category, levels = c("-2", "-1", "0", "1", "2"))) %>% 
  ggplot(aes(x = human_vs_model)) +
  geom_errorbar(aes(ymin=lower, ymax=upper, color=category), 
                position = position_dodge(.5), width = 0.3) +
  geom_point(aes(y = probability, color=category), position = position_dodge(.5),
             size = 3) +
  facet_grid(.~ condition) +
  xlab("Humans vs. LLMs") +
  ylab("Probability of informativeness rating") +
  theme(legend.position = "right", legend.title = element_blank(),
        axis.text = element_text(size = 16)) 
ggsave("info_exc4B.jpg", width = 7.00, height = 4.67, units = "in")

# trying out a better visualization after reviewer feedback
fitted_df %>% 
  mutate(condition = ifelse(condition == "FAR", "far", 
                            ifelse(condition == "SPLIT", "split", 
                                   ifelse(condition == "CLOSE", "close", NA)))) %>% 
  mutate(condition = factor(condition, levels = c("far", "split", "close"))) %>% 
  mutate(category = factor(category, levels = c("-2", "-1", "0", "1", "2"))) %>% 
  mutate(human_vs_model = ifelse(human_vs_model == "model", "llm", "human")) %>% 
  mutate(human_vs_model = factor(human_vs_model, levels = c("human", "llm"))) %>% 
  ggplot(aes(x = category)) +
  geom_errorbar(aes(ymin=lower, ymax=upper, color=human_vs_model), 
                position = position_dodge(.5), width = 0.3, size = 0.5) +
  geom_point(aes(y = probability, color=human_vs_model, shape=human_vs_model), position = position_dodge(.5),
             size = 3) +
  scale_shape_manual(values = c(16, 18)) +
  facet_grid(.~ condition) +
  xlab("Informativeness rating") +
  ylab("Probability of informativeness rating") +
  theme(legend.position = "right", legend.title = element_blank(),
        legend.text = element_text(size = 16),
        axis.text = element_text(size = 16)) +
  theme(panel.border = element_rect(color = "black", fill = "transparent"))
ggsave("info_exc4B_new.jpg", width = 7.00, height = 4.67, units = "in")

# model incl. all tested LLMs-----------------------------------------
fit_informativeness <- brm(informativeness_likert ~ human_vs_model * condition,
                           data = df,
                           family = cumulative(link = "probit"),
                           iter = 4000,
                           warmup = 1000,
                           cores = getOption("mc.cores", 4),
                           file = "fit_informativeness")
summary(fit_informativeness)
pp_check(fit_informativeness)

new_data <- data.frame(human_vs_model = rep(c("human", "model"), each=3),
                       condition = rep(c("SPLIT", "FAR", "CLOSE"), times=2))
# extract fitted values
fitted_probs <- fitted(fit_informativeness, newdata = new_data)

fitted_df <- data.frame(
  human_vs_model = rep(c("human", "model"), each=15),
  condition = rep(c("SPLIT", "FAR", "CLOSE"), each=5),
  category = rep(1:5, times = 6),
  probability = c(fitted_probs[1, "Estimate", ], fitted_probs[2, "Estimate",],
                  fitted_probs[3, "Estimate", ], fitted_probs[4, "Estimate", ],
                  fitted_probs[5, "Estimate", ], fitted_probs[6, "Estimate", ]),
  lower = c(fitted_probs[1, "Q2.5",], fitted_probs[2 , "Q2.5",],
            fitted_probs[3 , "Q2.5",], fitted_probs[4 , "Q2.5",],
            fitted_probs[5 , "Q2.5",], fitted_probs[6 , "Q2.5",]),
  upper = c(fitted_probs[1, "Q97.5",], fitted_probs[2 , "Q97.5",],
            fitted_probs[3 , "Q97.5",], fitted_probs[4 , "Q97.5",],
            fitted_probs[5 , "Q97.5",], fitted_probs[6 , "Q97.5",])
)

# transform ratings back to (-2, -1, 0, 1, 2)
fitted_df$category <- as.character(as.numeric(fitted_df$category) -3)

fitted_df %>% 
  mutate(condition = ifelse(condition == "FAR", "far", 
                            ifelse(condition == "SPLIT", "split", 
                                   ifelse(condition == "CLOSE", "close", NA)))) %>% 
  mutate(condition = factor(condition, levels = c("far", "split", "close"))) %>% 
  mutate(category = factor(category, levels = c("-2", "-1", "0", "1", "2"))) %>% 
  ggplot(aes(x = human_vs_model)) +
  geom_errorbar(aes(ymin=lower, ymax=upper, color=category), 
                position = position_dodge(.5), width = 0.3) +
  geom_point(aes(y = probability, color=category), position = position_dodge(.5),
             size = 3) +
  facet_grid(.~ condition) +
  xlab("Humans vs. LLMs") +
  ylab("Probability of informativeness rating") +
  theme(legend.position = "right", legend.title = element_blank(),
        axis.text = element_text(size = 16))
ggsave("info_all.jpg", width = 7.00, height = 4.67, units = "in")

# testing only the biggest model -------------------------------------
df_one <- df %>% filter(system == "Qwen3.5-27B-FP8_thinking" | system == "human")

fit_informativeness_one <- brm(informativeness_likert ~ cs(human_vs_model * condition),
                           data = df_one,
                           family = cumulative(link = "probit"),
                           iter = 4000,
                           warmup = 1000,
                           cores = getOption("mc.cores", 4),
                           file = "fit_informativeness_one")
summary(fit_informativeness_one)
pp_check(fit_informativeness_one)

new_data <- data.frame(human_vs_model = rep(c("human", "model"), each=3),
                       condition = rep(c("SPLIT", "FAR", "CLOSE"), times=2))
# extract fitted values
fitted_probs <- fitted(fit_informativeness_one, newdata = new_data)

fitted_df <- data.frame(
  human_vs_model = rep(c("human", "model"), each=15),
  condition = rep(c("SPLIT", "FAR", "CLOSE"), each=5),
  category = rep(1:5, times = 6),
  probability = c(fitted_probs[1, "Estimate", ], fitted_probs[2, "Estimate",],
                  fitted_probs[3, "Estimate", ], fitted_probs[4, "Estimate", ],
                  fitted_probs[5, "Estimate", ], fitted_probs[6, "Estimate", ]),
  lower = c(fitted_probs[1, "Q2.5",], fitted_probs[2 , "Q2.5",],
            fitted_probs[3 , "Q2.5",], fitted_probs[4 , "Q2.5",],
            fitted_probs[5 , "Q2.5",], fitted_probs[6 , "Q2.5",]),
  upper = c(fitted_probs[1, "Q97.5",], fitted_probs[2 , "Q97.5",],
            fitted_probs[3 , "Q97.5",], fitted_probs[4 , "Q97.5",],
            fitted_probs[5 , "Q97.5",], fitted_probs[6 , "Q97.5",])
)

# transform ratings back to (-2, -1, 0, 1, 2)
fitted_df$category <- as.character(as.numeric(fitted_df$category) -3)

fitted_df %>% 
  mutate(condition = ifelse(condition == "FAR", "far", 
                            ifelse(condition == "SPLIT", "split", 
                                   ifelse(condition == "CLOSE", "close", NA)))) %>% 
  mutate(condition = factor(condition, levels = c("far", "split", "close"))) %>% 
  mutate(category = factor(category, levels = c("-2", "-1", "0", "1", "2"))) %>% 
  ggplot(aes(x = human_vs_model)) +
  geom_errorbar(aes(ymin=lower, ymax=upper, color=category), 
                position = position_dodge(.5), width = 0.3) +
  geom_point(aes(y = probability, color=category), position = position_dodge(.5),
             size = 3) +
  facet_grid(.~ condition) +
  xlab("Humans vs. LLMs") +
  ylab("Probability of informativeness rating") +
  theme(legend.position = "right", legend.title = element_blank(),
        axis.text = element_text(size = 16))
ggsave("info_one.jpg", width = 7.00, height = 4.67, units = "in")
