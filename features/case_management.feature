Feature: Case Management System

@smoke
Scenario: User creates new case
  Given user logs in with username "manager@test.com" and password "Manager123!"
  When user clicks on "Create Case" button
  And user enters case number "CASE-2024-002"
  And user selects case type "Civil Litigation"
  And user enters case title "Smith vs Johnson"
  And user selects jurisdiction "New York"
  And user enters filing date "2024-03-20"
  And user clicks "Submit" button
  Then system displays message "Case created successfully"
  And case "CASE-2024-002" appears in active cases list
  And case status shows "Open"

@regression
Scenario: User assigns team members to case
  Given user navigates to case "CASE-2024-002"
  When user clicks on "Manage Team" button
  And user adds team member "john.doe@test.com" as "Lead Attorney"
  And user adds team member "jane.smith@test.com" as "Paralegal"
  And user clicks "Save Team" button
  Then system shows "2" team members for the case
  And notification is sent to added team members 