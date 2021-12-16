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
                      choices = c("N/A","Rot","Cage_Rot","Rot_post","Speed","Speed_post","LFP_aligned_stats",
                                  "Corr","firingRate","Corr-mvmt","lfp","Amph_prePost","Number_Of_Cells"),
                      selected = "N/A"),
        
       selectInput("y", 
                label = "Choose independent variable",
                choices = c("n/A"),
                selected = NULL),
       conditionalPanel(
         condition = "input.DataFile =='firingRate' ||input.DataFile == 'LFP_aligned_stats'",
         selectInput("cellType", "select cell type:",
                     list("all", "MSN", "CHI", "PV"))
       ),
       
       conditionalPanel(
         condition = "input.DataFile == 'Corr' ",
         selectInput("PopulationType", "select cell type:",
                     list("NA","MSN-MSN", "MSN-PV", "MSN-CHI"))
       ),
       conditionalPanel(
         condition = "input.DataFile == 'lfp'",
         selectInput("lfpMvmtType", "select mvmt type:",
                     list("NA","highC", "highAC", "lowRot", "hiSpeed","lowSpeed","rot","all"))
       ),
       conditionalPanel(
         condition = "input.DataFile == 'Amph_prePost'",
         selectInput("Amph", "period:",
                     list("NA","Healthy","Day 13-20","One Month"))
       ),
       conditionalPanel(
         condition = "input.DataFile == 'Corr-mvmt'",
         selectInput("mvmtType", "select mvmt type:",
                     list("NA","highContra", "highIpsi", "lowRot", "hiSpeed","lowSpeed","totRot"))
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
  require(svglite)
  require(qqplotr)
  
  #load the data sets
  dataCorr <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\bySessCompareCorr2Sig.csv")
  dataSpeed <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\speedData.csv")
  dataSpeedPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\lineraVelocity_post2.csv")
  dataRot <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\rotation_forR.csv")
  dataF <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\firingRateDf_R.csv")
  dataCorrMvmt <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\mvmtCorrSig.csv")
  datalfp <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\lfpForR.csv")
  dataRotPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\postAmphRot2R.csv")
  dataCageRot <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\rotation_count_forR.csv")
 # dataSpeedPost <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\postAmphSpeed2R.csv")
  dataAmphMvmt <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\prePostAmph_mvmtR.csv")
  dataNumN <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\numberNeurons.csv")
  dataPowerStat <- read.csv("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\paper1_edit5\\statsForPower.csv")
  

  
  values <- reactiveValues(data = dataRot)
  
  observeEvent(input$DataFile,ignoreInit = TRUE, {
    
    values$data <- switch(input$DataFile, "Rot" = dataRot,"Cage_Rot"=dataCageRot, "Rot_post"= dataRotPost, "Speed" = dataSpeed,
                          "Corr" = dataCorr,"Corr-mvmt" = dataCorrMvmt,"Speed_post" = dataSpeedPost,"LFP_aligned_stats" = dataPowerStat,
                          "firingRate" = dataF,"N/A" = dataRot,"lfp" = datalfp, "Amph_prePost" = dataAmphMvmt,"Number_Of_Cells" =dataNumN)
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
  
  
  observeEvent(input$Amph,ignoreInit = TRUE, {
    if (input$Amph != 'NA' ) {
      m = ""
      for (per in unique(values$data$Period)){
        df <- values$data %>% filter(mvmt == input$y & Period == per)
        x <- df$time[df$drug =="pre"]
        y <- df$time[df$drug =="post"]
        res <- wilcox.test(x, y, paired = TRUE)
        m <- paste(m,sprintf("%s: p-Value = %f, from %0.2f%% to %0.2f%%",per,res$p.value,mean(x)*100,mean(y)*100),sep ='\n')
      }
      df <- values$data %>% filter(mvmt == input$y )
      df <- reshape(df, direction="wide", idvar=c("sess", "Period","mvmt"), timevar="drug")
      df <- df %>% 
        rename(
          post_Amph = time.post,
          pre_Amph = time.pre
        )
      output$glmm <- renderPrint( print(cat(m)))
      p <- ggpaired(df, cond1 = "pre_Amph", cond2 = "post_Amph",
                    color = "condition", palette = "jco", 
                    line.color = "gray", line.size = 0.4,
                    facet.by = "Period", short.panel.labs = FALSE)+
           stat_compare_means(label = "p.format", paired = TRUE)
      # Use only p.format as label. Remove method name.
      output$plot1 <- renderPlot(p )
      ggsave(file=sprintf("C:\\Users\\zemel\\Documents\\Reaserch\\6OHDA\\figs\\R_plots\\%s.svg",input$y), plot=p, width=10, height=8)
      #output$plot1 <- renderPlot( boxplot(time ~ drug*Period , data=df,
      #                                    col=(c("gold","darkgreen"))))
    }
  })
  
  
  
  
  
}

# Run the application 
shinyApp(ui = ui, server = server)
