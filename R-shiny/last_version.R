library(shiny)
library(shinyjs)
library(reticulate)
library(DT)
library(plotly)
library(ggplot2)
library(jsonlite)

# Utilisation de Python via reticulate
use_python("C:/ProgramData/anaconda3/python.exe", required = TRUE)

# Chargement des scripts Python
source_python("C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/SMLibrary/sp_function.py")
source_python("C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/SMLibrary/pr.py")
source_python("C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/SMLibrary/bc.py")
source_python("C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/SMLibrary/models_2.py")
source_python("C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/SMLibrary/watcher.py")


# Augmentation de la taille maximale des requ??tes Shiny
options(shiny.maxRequestSize = 10000 * 1024^2)

# Interface utilisateur
ui <- fluidPage(
  useShinyjs(),
  tags$link(rel = "stylesheet", href = "https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"),
  tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css"),
  tags$link(rel = "stylesheet", href = "https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css"), tags$style(HTML("
    .navbar { background-color: #7C93C3 !important; }
    .navbar-nav { display: flex; flex-direction: row; }
    .nav-item { margin-right: 20px; }
    .nav-link { color: white !important; }
    .navbar-brand { color: white !important; margin-right: 20px; }
    .logo { height: 40px; margin-right: 10px; }
    .action-button-blue { background-color: #55679C; color: #ffffff; border-color: #55679C; }
    .action-button-blue:hover { background-color: #0056b3; color: #ffffff; border-color: #0056b3; }
    .icon-container { text-align: center; margin: 40px; }
    .icon { font-size: 50px; color: #007bff; margin: 10px; }
    .section { margin-bottom: 30px; }
    .section h3 { color: #7C93C3; }
  ")),
  
  tags$style(HTML("
    body {
      background-color: #FFFFFF; 
      color: #55679C;
    }

    .sidebar-panel {
      background-color: #A0C4E2; /* Bleu Pastel Clair */
      color: #4F4F4F; /* Gris Anthracite */
      border-right: 2px solid #4F4F4F; /* Bordure droite */
      padding: 20px;
    }

    .main-panel {
      background-color:  #7C93C3;
      color: #7C93C3;
      padding: 20px;
    }
    
    hr {
      border: 2x solid #C0C0C0; 
    }
  ")),
  
  navbarPage("Profiler",
             tabPanel("Home",
                      fluidPage(
                        titlePanel("Welcome to the Profiler Application"),
                        div(class = "section",
                            h3("Overview"),
                            p("This application allows for comprehensive data analysis and classification for mass spectrometry data. You can convert RAW files to mzML, load and preprocess data, train models, and classify data in real-time.")
                        ),
                        div(class = "section",
                            h3("Data Analysis"),
                            p("In the Data Analysis tab, you can perform the following tasks:"),
                            tags$ul(
                              tags$li("Convert RAW files to mzML format."),
                              tags$li("Load and preprocess your data."),
                              tags$li("Train and evaluate machine learning models.")
                            ),
                        ),
                        div(class = "section",
                            h3("Real-Time Diagnosis"),
                            p("In the Real-Time Diagnosis tab, you can:"),
                            tags$ul(
                              tags$li("Load trained models and their features."),
                              tags$li("Monitor a directory for new data and classify it in real-time.")
                            ),
                        ),
                        div(class = "section",
                            h3("Help and Documentation"),
                            p("Need help? Access the documentation and tutorials to get started with the Profiler application."),
                        )
                      )
             ),
             tabPanel("Data Analysis",
                      fluidPage(
                        sidebarLayout(
                          sidebarPanel(
                            h3("Conversion"),
                            textInput("raw_files_path", "Enter Path for RAW Files Directory", value = ""),
                            textInput("mzml_output_dir", "Enter Output Directory for mzML", value = ""),
                            actionButton("convert_button", "Convert RAW to mzML", class = "action-button-blue"),
                            textOutput("conversionResult"),
                            verbatimTextOutput("debugConversion"),
                            hr(),
                            
                            h3("Spectres Moyens"),
                            actionButton("plot_mean_spectra", "Afficher les Spectres Moyens", class = "action-button-blue"),
                            hr(),
                            
                            h3("Histogramme des Classes"),
                            actionButton("plot_class_histogram", "Afficher l'Histogramme des Classes", class = "action-button-blue"),
                            hr(),
                            
                            h3("Data Loading"),
                            uiOutput("paths_ui"),
                            actionButton("add_path", "Add Path", class = "action-button-blue"),
                            actionButton("load_data", "Load Data", class = "action-button-blue"),
                            tags$hr(),
                            
                            h3("Data Preprocessing"),
                            actionButton("preprocess_data", "Preprocess Data", class = "action-button-blue"),
                            hr(),
                            
                            h3("Train and Evaluate Models"),
                            numericInput("n_components", "Number of SVD components", value = 50),
                            selectInput("balance_method", "Balance Method", choices = c("None", "SMOTE", "ADASYN"), selected = "None"),
                            actionButton("train_models", "Train and Evaluate Models", class = "action-button-blue"),
                            hr(),
                            
                            h3("Save Data"),
                            fileInput("save_csv", "Save CSV Path", accept = c(".csv")),
                            actionButton("save_data", "Save DataFrame", class = "action-button-blue"),
                            
                            h3("Save Model"),
                            selectInput("model_name", "Select Model to Save", choices = NULL),
                            textInput("model_save_folder", "Select Folder to Save Model", value = ""),
                            actionButton("save_model", "Save Model", class = "action-button-blue")
                          ),
                          mainPanel(
                            tabsetPanel(
                              tabPanel("Conversion Result", textOutput("conversionResult")),
                              tabPanel("Loaded Data", tableOutput("loaded_data")),
                              tabPanel("Preprocessed Data", DTOutput("preprocessed_data_summary")),
                              tabPanel("Resampled Data", DTOutput("resampled_data")),
                              tabPanel("Model Scores", uiOutput("model_scores")),
                              tabPanel("Confusion Matrices", uiOutput("confusion_matrices")),
                              tabPanel("mzML Files", verbatimTextOutput("mzml_files")),
                              tabPanel("Spectres Moyens", plotOutput("mean_spectra_plot")),
                              tabPanel("Histogramme des Classes", plotOutput("class_histogram_plot"))
                            )
                          )
                        )
                      )
             ),
             tabPanel("Real-Time Diagnosis",
                      sidebarLayout(
                        sidebarPanel(
                          textInput("watch_directory", "Watch Directory", value = ""),
                          textInput("output_directory", "Output Directory", value = ""),
                          actionButton("load_button", "Load Models and Features", class = "action-button-blue"),
                          actionButton("start_button", "Start Watcher", class = "action-button-blue"),
                          actionButton("stop_button", "Stop Watcher", class = "action-button-blue"),
                          verbatimTextOutput("status")
                        ),
                        mainPanel(
                          h3("Model Predictions"),
                          tableOutput("prediction_table")
                        )
                      )
             )
  )
)
server <- function(input, output, session) {
  
  rv <- reactiveValues(data = NULL, preprocessed_data = NULL, resampled_data = NULL, pipelines = NULL)
  
  showLoadingModal <- function(message) {
    showModal(modalDialog(
      title = "Please wait",
      h5(paste0(message, "...")),
      footer = NULL,
      easyClose = FALSE
    ))
  }
  
  observeEvent(input$convert_button, {
    req(input$raw_files_path, input$mzml_output_dir)
    
    raw_files_path <- input$raw_files_path
    mzml_output_dir <- input$mzml_output_dir
    
    if (raw_files_path == "" || mzml_output_dir == "") {
      showModal(modalDialog(
        title = "Error",
        "Please specify both the path for RAW files and the output directory for mzML.",
        easyClose = TRUE
      ))
      return(NULL)
    }
    
    showLoadingModal("Starting RAW to mzML conversion")
    
    tryCatch({
      convert_raw_to_mzml(raw_files_path, mzml_output_dir)
      removeModal()
      output$conversionResult <- renderText("Conversion completed successfully!")
      output$debugConversion <- renderPrint({
        cat("Conversion completed successfully!\n")
      })
    }, error = function(e) {
      removeModal()
      output$conversionResult <- renderText(paste("Error during RAW to mzML conversion:", e$message))
      output$debugConversion <- renderPrint({
        cat("Error during RAW to mzML conversion:", e$message, "\n")
      })
    })
  })
  
  observeEvent(input$add_path, {
    num_paths <- length(grep("^path", names(input)))
    insertUI(
      selector = "#paths_ui",
      ui = tagList(
        textInput(paste0("path", num_paths + 1), "Path to .mzML Directory", placeholder = "Enter path here"),
        textInput(paste0("class", num_paths + 1), "Class Name for this Directory", placeholder = "Enter class name here")
      )
    )
  })
  
  observeEvent(input$load_data, {
    num_paths <- length(grep("^path", names(input)))
    paths <- sapply(1:num_paths, function(i) input[[paste0("path", i)]])
    classes <- sapply(1:num_paths, function(i) input[[paste0("class", i)]])
    
    paths <- paths[!is.null(paths) & paths != ""]
    classes <- classes[!is.null(classes) & classes != ""]
    
    if (length(paths) == 0 || length(classes) == 0 || length(paths) != length(classes)) {
      showModal(modalDialog(
        title = "Error",
        "Please fill in all path and class fields.",
        easyClose = TRUE
      ))
      return(NULL)
    }
    
    showLoadingModal("Loading data")
    
    tryCatch({
      df <- load_data(paths, classes)
      rv$data <- df
      removeModal()
      output$loaded_data <- renderTable({
        head(df)
      })
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("An error occurred:", e$message),
        easyClose = TRUE
      ))
    })
  })
  
  observeEvent(input$preprocess_data, {
    req(rv$data)
    
    showLoadingModal("Preprocessing data")
    
    tryCatch({
      df <- preprocess_data(rv$data)
      rv$preprocessed_data <- df
      output$preprocessed_data_summary <- renderDT({
        datatable(df)
      })
      removeModal()
      showModal(modalDialog(
        title = "Success",
        "Data has been preprocessed successfully!",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Error during data preprocessing:", e$message),
        easyClose = TRUE
      ))
    })
  })
  
  observeEvent(input$plot_class_histogram, {
    req(rv$data)  # V??rifie que les donn??es ont ??t?? charg??es
    
    # Cr??ation de l'histogramme bas?? sur la colonne 'Class'
    output$class_histogram_plot <- renderPlot({
      ggplot(rv$data, aes(x = factor(Class))) +  # Utilisation de la colonne 'Class'
        geom_bar(fill = "steelblue") +
        labs(title = "Distribution des Classes", x = "Classe", y = "Nombre d'??chantillons") +
        theme_minimal()
    })
  })
  # Calcul et affichage des spectres moyens
  output$mean_spectra_plot <- renderPlot({
    req(rv$data) # Assurez-vous que les donn??es sont disponibles
    
    # S??lection des colonnes spectrales
    spectral_data <- rv$preprocessed_data [, -(1:3)] # Supposons que les 3 premi??res colonnes ne sont pas des spectres
    
    # Calculer le spectre moyen par classe
    mean_spectra <- aggregate(spectral_data, by = list(rv$preprocessed_data$Class), FUN = mean)
    class_labels <- mean_spectra[, 1]
    mean_spectra <- mean_spectra[, -1]
    
    # Cr??er un graphique pour chaque classe
    plots <- list()
    for (i in seq_along(class_labels)) {
      plot <- ggplot() +
        geom_line(aes(x = 1:ncol(mean_spectra), y = as.numeric(mean_spectra[i, ])), color = "blue") +
        labs(title = paste("Spectre moyen pour la classe", class_labels[i]),
             x = "M/z",
             y = "Intensit?? moyenne") +
        theme_minimal()
      plots[[i]] <- plot
    }
    
    # Afficher tous les graphiques dans une seule fen??tre
    gridExtra::grid.arrange(grobs = plots, ncol = 1)
  })
  
  observeEvent(input$train_models, {
    req(rv$preprocessed_data)
    
    showLoadingModal("Training and evaluating models")
    
    tryCatch({
      if (input$balance_method == "SMOTE") {
        df_resampled <- apply_smote(rv$preprocessed_data)
      } else if (input$balance_method == "ADASYN") {
        df_resampled <- apply_adasyn(rv$preprocessed_data)
      } else {
        df_resampled <- rv$preprocessed_data
      }
      
      output_dir <- 'C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/matrice_confusion_output'
      dir.create(output_dir, showWarnings = FALSE)
      
      results <- train_and_evaluate_pipeline(df_resampled, input$n_components, output_dir)
      
      rv$pipelines <- results[[1]]
      
      output$model_scores <- renderUI({
        lapply(names(rv$pipelines), function(model_name) {
          model_info <- results[[3]][[model_name]]
          model_report <- model_info$classification_report
          model_scores <- model_info$mean_score
          model_std <- model_info$std_score
          
          div(
            h4(paste("Model:", model_name)),
            verbatimTextOutput(paste0("report_", model_name)),
            p(paste("Mean score:", model_scores)),
            p(paste("Standard deviation:", model_std))
          )
        })
      })
      
      lapply(names(rv$pipelines), function(model_name) {
        model_info <- results[[3]][[model_name]]
        model_report <- model_info$classification_report
        
        output[[paste0("report_", model_name)]] <- renderPrint({
          cat(model_report)
        })
      })
      
      output$confusion_matrices <- renderUI({
        output_dir <- 'C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/matrice_confusion_output'
        
        model_files <- list.files(output_dir, pattern = "*_confusion_matrix.json", full.names = TRUE)
        
        lapply(model_files, function(json_file) {
          model_name <- basename(json_file)
          model_name <- gsub("_confusion_matrix.json", "", model_name)
          
          plotlyOutput(outputId = paste0("plot_", model_name))
        })
      })
      
      lapply(list.files('C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/matrice_confusion_output', pattern = "*_confusion_matrix.json", full.names = TRUE), function(json_file) {
        model_name <- gsub("_confusion_matrix.json", "", basename(json_file))
        
        output[[paste0("plot_", model_name)]] <- renderPlotly({
          tryCatch({
            conf_matrix <- fromJSON(json_file)
            plot_ly(
              x = conf_matrix$data[[1]]$x,
              y = conf_matrix$data[[1]]$y,
              z = conf_matrix$data[[1]]$z,
              type = "heatmap",
              colorscale = "Viridis"
            ) %>%
              layout(
                xaxis = list(title = "Predicted Label"),
                yaxis = list(title = "True Label"),
                title = paste("Confusion Matrix for Model:", model_name)
              )
          }, error = function(e) {
            print(paste("Error loading JSON file:", json_file))
            print(e)
            NULL
          })
        })
      })
      
      removeModal()
      showModal(modalDialog(
        title = "Success",
        "Models have been trained and evaluated successfully!",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Error during model training and evaluation:", e$message),
        easyClose = TRUE
      ))
    })
  })
  
  observeEvent(input$save_data, {
    req(rv$preprocessed_data)
    
    showLoadingModal("Saving DataFrame")
    
    tryCatch({
      save_path <- input$save_csv$datapath
      write.csv(rv$preprocessed_data, save_path, row.names = FALSE)
      removeModal()
      showModal(modalDialog(
        title = "Success",
        "DataFrame saved successfully!",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Error saving DataFrame:", e$message),
        easyClose = TRUE
      ))
    })
  })
  
  observeEvent(input$save_model, {
    req(rv$pipelines)
    
    showLoadingModal("Saving Model")
    
    tryCatch({
      model_path <- input$model_save$datapath
      save_selected_models(rv$pipelines, input$model_name, dirname(model_path))
      removeModal()
      showModal(modalDialog(
        title = "Success",
        "Model saved successfully!",
        easyClose = TRUE
      ))
    }, error = function(e) {
      removeModal()
      showModal(modalDialog(
        title = "Error",
        paste("Error saving model:", e$message),
        easyClose = TRUE
      ))
    })
  })
  
  output$mzml_files <- renderPrint({
    mzml_files <- list.files(input$mzml_output_dir, pattern = "\\.mzML$", full.names = TRUE)
    mzml_files
  })
  
  # Real-Time Classification - Load models and numeric features
  observeEvent(input$load_button, {
    model_paths <- c(
      'C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/HistGradientBoostingClassifier_last.pkl',
      'C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/RandomForestClassifier_last.pkl'
    )
    nf_path <- 'C:/Users/Admin/Documents/Ikram_Rshiny/R shiny/numeric_features.pkl'
    
    # Call Python function to load models and numeric features
    py$load_models_and_features(model_paths, nf_path)
    output$status <- renderText("Models and features loaded.")
  })
  
  # Start real-time watcher
  observeEvent(input$start_button, {
    req(input$watch_directory, input$output_directory)
    py$start_watcher(input$watch_directory, input$output_directory)
    output$status <- renderText("Watcher started.")
  })
  
  # Stop real-time watcher
  observeEvent(input$stop_button, {
    py$stop_watcher()
    output$status <- renderText("Watcher stopped.")
  })
  
}

shinyApp(ui, server)