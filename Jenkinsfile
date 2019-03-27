#!/usr/bin/env groovy

pipeline {
    agent none
    stages {
        stage('Build and Test') {
            parallel {
                stage('Linux') {
                    agent {
                        kubernetes {
                            label 'jnlp-pod'
                            defaultContainer 'petsc'
                            yamlFile '.kubernetes-pod.yaml'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                echo "Current dir: ${pwd()}"
                                sh "make"
                            }
                        }
                        stage('Test') {
                            steps {
                                sh "make junit"
                            }
                            post {
                                always {
                                    junit "**/build/*.junit"
                                    sh "make clean"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
    
