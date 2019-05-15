## Introduction of the project

Git is a Version Control System that was first released in 2005 and has grown to be a popular tool used across multiple domains. As distributed as it is, Git still remains to be somehow difficult to learn. A lot of users encounter some common Git problems every day while using it, such as merge conflicts, pushing the incorrect changes, committing to the wrong branches, etc..

We have the hypothesis that there are common Git workflows, and these workflows account for a large fraction of everyday use. RStudio is interested in developing a new tool for Git users that supports “common” workflows, and only these workflows.

Our project aims to identify these common workflows, with the eventual goal of providing recommendations for what features should and should not be included in an easy-to-use alternative to Git.

Two main questions we are trying to understand:
- What are the Git workflow patterns that are being used most widely? With this question we want to see if we can confirm that users follow workflows such as the Git Flow or if they follow other common workflows that are more intuitive for them.
- What are the patterns that, while possible to perform using Git, are not being used widely? This question will enable us to understand how different workflows are used in different contexts.

By answering this two questions, we will gain insights that will enable the development of a new tool that improves and consolidates workflows for users of Version Control Systems.

We visualize collections of Git commits as graphs using NetworkX, a popular Python package for analyzing and visualizing complex networks. Git commits form a directed acyclic graph (DAG) so we are visualizing them as such. Every commit is a node, and every parent-child relationship between commits is a directed edge. A node with 2 children occurs when there is a branch, and a node with 2 parents occurs when there is a merge.

![](/img/posts/git_repo_1.png) 
*Figure: Example of Github repositories shown in NetworkX graph*
