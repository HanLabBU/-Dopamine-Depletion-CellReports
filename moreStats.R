#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
ui <- fluidPage(
    
    # Application title
    titlePanel("GLMM diagnostics"),
    
    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(position = "right",
                     selectInput("DataFile", 
                                 label = "Choose dataset:",
                                 choices = c("N/A","Corr","Corr-mvmt","Speed-Saline","post-drug-corr","lfp","Gamma"),
                                 # choices = c("N/A","Rot","Cage_Rot","Rot_post","Speed","Speed_post","LFP_aligned_stats",
                                 #             "Corr","firingRate","Corr-mvmt","Amph_prePost","Number_Of_Cells"),
                                 selected = "N/A"),
                     
                     selectInput("y", 
                                 label = "Choose independent variable",
                                 choices = c("n/A"),
                                 selected = NULL),
                     conditionalPanel(
                         condition = "input.DataFile =='firingRate' ||input.DataFile == 'Gamma'",
                         selectInput("cellType", "select cell type:",
                                     list("mvmt", "MSN", "CHI", "PV"))
                     ),
                     
                     conditionalPanel(
                         condition = "input.DataFile == 'Corr' || input.DataFile=='post-drug-corr'",
                         selectInput("PopulationType", "select cell type:",
                                     list("NA","MSN-MSN", "MSN-PV", "MSN-CHI"))
                     ),
                     conditionalPanel(
                         condition = "input.DataFile == 'lfp'",
                         selectInput("lfpMvmtType", "select mvmt type:",
                                     list("NA","highC", "highAC", "lowRot", "hiSpeed","lowSpeed","rot","all"))
                     ),
                     conditionalPanel(
                         condition = "input.DataFile == 'post-drug-corr'",
                         selectInput("drug", "Drug:",
                                     list("NA","L-Dopa","Amphetamine","Saline"))
                     ),
                     conditionalPanel(
                         condition = "input.DataFile == 'Corr-mvmt'",
                         selectInput("mvmtType", "select mvmt type:",
                                     list("NA","highContra", "highIpsi", "lowRot", "hiSpeed","lowSpeed","totRot","all"))
                     ),
                     conditionalPanel(
                         condition = "input.DataFile == 'Corr-mvmt'",
                         selectInput("population", "select cell type:",
                                     list("NA","MSN-MSN", "MSN-PV", "MSN-CHI"))
                     )),
        
        # Show a plot of the generated distribution
        mainPanel(
            
            verbatimTextOutput("glmm"),
            verbatimTextOutput("anova"),
            plotOutput("mouseQQ"),
            plotOutput("plot1"),
            plotOutput("risidualPlot")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output,session) {
    require(lme4)
    require(lmerTest)
    require(dplyr)
    require(ggpubr)
    require(svglite)
    require(qqplotr)
    
    #load the data sets
    dataCorr <- read.csv("D:\\6OHDA\\submission2\\bySessCompareCorr2Sig.csv")
    dataCorrMvmt <- read.csv("D:\\6OHDA\\submission2\\mvmtCorrSig.csv")
    dataCorrPrePost <- read.csv("D:\\6OHDA\\submission2\\bySessCompareCorr2_post.csv")
    #dataCorr <- read.csv("D:\\6OHDA\\submission2\\bySessCompareCorr2Sig3.csv")
    #dataCorrMvmt <- read.csv("D:\\6OHDA\\submission2\\mvmtCorrSig.csv")
    dataSpeed <- read.csv("D:\\6OHDA\\submission2\\saline_control.csv")
    datalfp <- read.csv("D:\\6OHDA\\submission2\\lfpForR.csv")
    dataGamma <- read.csv("D:\\6OHDA\\submission2\\postHighGammaStats.csv")
    
    
    # dataSpeed <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\speedData.csv")
    # dataSpeedPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\lineraVelocity_post2.csv")
    # dataRot <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\rotation_forR.csv")
    # dataF <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\firingRateDf_R.csv")
    # datalfp <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\lfpForR.csv")
    # dataRotPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\postAmphRot2R.csv")
    # dataCageRot <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\rotation_count_forR.csv")
    # # dataSpeedPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\postAmphSpeed2R.csv")
    # dataAmphMvmt <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\prePostAmph_mvmtR.csv")
    # dataNumN <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\numberNeurons.csv")
    # dataPowerStat <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\statsForPower.csv")
    
    
    
    values <- reactiveValues(data = dataCorr)
    
    observeEvent(input$DataFile,ignoreInit = TRUE, {
        
    values$data <- switch(input$DataFile, "Corr" = dataCorr,"Corr-mvmt" = dataCorrMvmt,
                          "Speed-Saline" = dataSpeed,"post-drug-corr"=dataCorrPrePost,"lfp" = datalfp,"Gamma"=dataGamma)

        # values$data <- switch(input$DataFile, "Rot" = dataRot,"Cage_Rot"=dataCageRot, "Rot_post"= dataRotPost, "Speed" = dataSpeed,
        #                       "Corr" = dataCorr,"Corr-mvmt" = dataCorrMvmt,"Speed_post" = dataSpeedPost,"LFP_aligned_stats" = dataPowerStat,
        #                       "firingRate" = dataF,"N/A" = dataRot,"lfp" = datalfp, "Amph_prePost" = dataAmphMvmt,"Number_Of_Cells" =dataNumN)
        # relevel the data set
        values$data$Period <- as.factor(values$data$Period)
        values$data$Period <- relevel(values$data$Period, ref="Healthy")
        #values$data <- values$data %>% filter(!Mouse %in% c("1253","1231",1253,1231))
        
        if (input$DataFile == "Amph_prePost"){
            updateSelectInput(session, "y", choices = unique(values$data$mvmt),selected = 'low_speed')
        }else{
            updateSelectInput(session, "y", choices = colnames(values$data),selected = 'Period')}
    })
    observeEvent(input$y,ignoreInit = TRUE, {
        if (input$y != 'Period' & input$DataFile != "Amph_prePost") {
            glmm <- lmer(paste0(input$y,' ~ Period + (1|Mouse)',sep=''),
                         data=values$data)
            output$glmm <- renderPrint( print(summary(glmm)))
            output$anova <- renderPrint( print(anova(glmm)))
            output$plot1 <- renderPlot(plot(glmm))
            output$risidualPlot <- renderPlot({qqnorm(resid(glmm))
                qqline(resid(glmm))})
            bmat <- as.data.frame(lme4::ranef(glmm))
            renest <- tidyr::nest(bmat, data = c("grp", "condval", "condsd"))
            plots <- purrr::pmap(list(renest$data, renest$grpvar, renest$term),
                                 function(a,b,c){
                                     ggplot(data = a, aes_string(sample = "condval")) +
                                         qqplotr::stat_qq_band(bandType = "pointwise",
                                                               distribution = "norm",
                                                               fill = "#FBB4AE", alpha = 0.4) +
                                         qqplotr::stat_qq_line(distribution = "norm", colour = "#FBB4AE") +
                                         qqplotr::stat_qq_point(distribution = "norm") +
                                         xlab("Normal quantiles") + theme_bw() +
                                         ylab(paste(b,": ", c))
                                 }
            )
            output$mouseQQ <- renderPlot(cowplot::plot_grid(plotlist = plots))
        }
    })
    observeEvent(input$cellType,ignoreInit = TRUE, {
        if (input$cellType != 'all' & input$y != 'Period') {
            glmm <- lmer(paste0(input$y,' ~ Period + (1|Mouse) ',sep=''),
                         data=values$data %>% filter(CellType == input$cellType))
            output$glmm <- renderPrint( print(summary(glmm)))
            output$anova <- renderPrint( print(anova(glmm)))
            output$plot1 <- renderPlot(plot(glmm))
            output$risidualPlot <- renderPlot({qqnorm(resid(glmm))
                qqline(resid(glmm))})
            bmat <- as.data.frame(lme4::ranef(glmm))
            renest <- tidyr::nest(bmat, data = c("grp", "condval", "condsd"))
            plots <- purrr::pmap(list(renest$data, renest$grpvar, renest$term),
                                 function(a,b,c){
                                     ggplot(data = a, aes_string(sample = "condval")) +
                                         qqplotr::stat_qq_band(bandType = "pointwise",
                                                               distribution = "norm",
                                                               fill = "#FBB4AE", alpha = 0.4) +
                                         qqplotr::stat_qq_line(distribution = "norm", colour = "#FBB4AE") +
                                         qqplotr::stat_qq_point(distribution = "norm") +
                                         xlab("Normal quantiles") + theme_bw() +
                                         ylab(paste(b,": ", c))
                                 }
            )
            output$mouseQQ <- renderPlot(cowplot::plot_grid(plotlist = plots))
            
        }
    })
    observeEvent(input$PopulationType,ignoreInit = TRUE, {
        if (input$PopulationType != 'NA' & input$y != 'Period') {
            glmm <- lmer(paste0(input$y,' ~ Period + (1|Mouse) ',sep=''),
                         data=values$data %>% group_by('Sess','Period','population') %>% filter(population == input$PopulationType))
            output$glmm <- renderPrint( print(summary(glmm)))
            output$anova <- renderPrint( print(anova(glmm)))
            output$plot1 <- renderPlot(plot(glmm))
            output$risidualPlot <- renderPlot({qqnorm(resid(glmm))
                qqline(resid(glmm))})
            bmat <- as.data.frame(lme4::ranef(glmm))
            renest <- tidyr::nest(bmat, data = c("grp", "condval", "condsd"))
            plots <- purrr::pmap(list(renest$data, renest$grpvar, renest$term),
                                 function(a,b,c){
                                     ggplot(data = a, aes_string(sample = "condval")) +
                                         qqplotr::stat_qq_band(bandType = "pointwise",
                                                               distribution = "norm",
                                                               fill = "#FBB4AE", alpha = 0.4) +
                                         qqplotr::stat_qq_line(distribution = "norm", colour = "#FBB4AE") +
                                         qqplotr::stat_qq_point(distribution = "norm") +
                                         xlab("Normal quantiles") + theme_bw() +
                                         ylab(paste(b,": ", c))
                                 }
            )
            output$mouseQQ <- renderPlot(cowplot::plot_grid(plotlist = plots))
            
        }
    })
    observeEvent(input$lfpMvmtType,ignoreInit = TRUE, {
        if (input$lfpMvmtType != 'NA' & input$y != 'Period') {
            glmm <- lmer(paste0(input$y,' ~ Period + (1|Mouse) ',sep=''),
                         data=values$data %>% filter(mvmt == input$lfpMvmtType))
            output$glmm <- renderPrint( print(summary(glmm)))
            output$anova <- renderPrint( print(anova(glmm)))
            output$plot1 <- renderPlot(plot(glmm))
            output$risidualPlot <- renderPlot({qqnorm(resid(glmm))
                qqline(resid(glmm))})
            bmat <- as.data.frame(lme4::ranef(glmm))
            renest <- tidyr::nest(bmat, data = c("grp", "condval", "condsd"))
            plots <- purrr::pmap(list(renest$data, renest$grpvar, renest$term),
                                 function(a,b,c){
                                     ggplot(data = a, aes_string(sample = "condval")) +
                                         qqplotr::stat_qq_band(bandType = "pointwise",
                                                               distribution = "norm",
                                                               fill = "#FBB4AE", alpha = 0.4) +
                                         qqplotr::stat_qq_line(distribution = "norm", colour = "#FBB4AE") +
                                         qqplotr::stat_qq_point(distribution = "norm") +
                                         xlab("Normal quantiles") + theme_bw() +
                                         ylab(paste(b,": ", c))
                                 }
            )
            output$mouseQQ <- renderPlot(cowplot::plot_grid(plotlist = plots))
            
        }
    })
    observeEvent(input$population,ignoreInit = TRUE, {
        if (input$population != 'NA' & input$y != 'Period') {
            glmm <- reactive({
                lmer(paste0(input$y,' ~ Period + (1|Mouse)',sep=''),
                     data=values$data %>% filter(mvmt == input$mvmtType & population ==input$population ))
            }) 
            output$glmm <- renderPrint( print(summary(glmm())))
            output$anova <- renderPrint( print(anova(glmm())))
            output$plot1 <- renderPlot(plot(glmm()))
            output$risidualPlot <- renderPlot({qqnorm(resid(glmm()))
                qqline(resid(glmm()))})
            bmat <- as.data.frame(lme4::ranef(glmm()))
            renest <- tidyr::nest(bmat, data = c("grp", "condval", "condsd"))
            plots <- purrr::pmap(list(renest$data, renest$grpvar, renest$term),
                                 function(a,b,c){
                                     ggplot(data = a, aes_string(sample = "condval")) +
                                         qqplotr::stat_qq_band(bandType = "pointwise",
                                                               distribution = "norm",
                                                               fill = "#FBB4AE", alpha = 0.4) +
                                         qqplotr::stat_qq_line(distribution = "norm", colour = "#FBB4AE") +
                                         qqplotr::stat_qq_point(distribution = "norm") +
                                         xlab("Normal quantiles") + theme_bw() +
                                         ylab(paste(b,": ", c))
                                 }
            )
            output$mouseQQ <- renderPlot(cowplot::plot_grid(plotlist = plots))
            
        }
        
    })
    
    
    observeEvent(input$drug,ignoreInit = TRUE, {
        if (input$drug != 'NA' & input$PopulationType != 'NA' & input$y != 'Period') {
            m = ""
            for (per in unique(values$data$Period)){
                df <- values$data %>% filter(population == input$PopulationType, Period == per, Drug==input$drug)
                x <- df[df$timeRange =="Pre",input$y]
                y <- df[df$timeRange =="Post",input$y]
                res <- wilcox.test(x, y, paired = TRUE)
                m <- paste(m,sprintf(" %s - %s: p-Value = %f, from %0.2f to %0.2f",input$drug,per,res$p.value,mean(x),mean(y)),sep ='\n')
            }
            df <- values$data %>% filter(population == input$PopulationType, Drug==input$drug)%>% rename(val = input$y)
            df <- reshape(df, direction="wide", idvar=c("Sess", "Period"), timevar="timeRange")
            df <- df %>% 
                rename(
                    post_drug = val.Post,
                    pre_drug = val.Pre
                )
            output$glmm <- renderPrint( print(cat(m)))
            p <- ggpaired(df, cond1 = "pre_drug", cond2 = "post_drug",
                          color = "condition", palette = "jco", 
                          line.color = "gray", line.size = 0.4,
                          facet.by = "Period", short.panel.labs = FALSE)+
                stat_compare_means(label = "p.format", paired = TRUE)
            # Use only p.format as label. Remove method name.
            output$plot1 <- renderPlot(p )
            ggsave(file=sprintf("D:\\extraAnalysis\\figs\\%s_%s.svg",input$y,input$PopulationType), plot=p, width=10, height=8)
            #output$plot1 <- renderPlot( boxplot(time ~ drug*Period , data=df,
            #                                    col=(c("gold","darkgreen"))))
        }
    })
    
    
    
    
    
}

# Run the application 
shinyApp(ui = ui, server = server)
