Feature: Document Review System

@regression
Scenario: User performs document review
  Given user logs in with username "reviewer@test.com" and password "Review123!"
  When user opens document "Important Contract"
  And user adds comment "Section 3.1 needs revision" at page "2"
  And user highlights text "payment terms" in yellow
  And user marks document status as "Needs Revision"
  And user assigns review to "legal.team@test.com"
  Then system saves review timestamp
  And notification is sent to assigned reviewer

@sanity
Scenario Outline: User applies different review statuses
  Given user has opened document for review
  When user selects review status "<status>"
  And user adds mandatory comments for "<status>"
  Then system updates document status to "<status>"
  And appropriate workflow is triggered for "<status>"

  Examples:
    | status          |
    | Approved        |
    | Needs Revision  |
    | Rejected        | 