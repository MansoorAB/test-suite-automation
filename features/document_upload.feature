Feature: Document Upload Feature

@smoke @regression
Scenario: Upload a valid document
  Given the user is on the "document upload" page
  When the user selects the file "valid_document.pdf"
  And the user clicks the "upload" button
  Then the system should display the message "Upload successful"
  And the document "valid_document.pdf" should be listed in the user's documents

@smoke @regression
Scenario: Attempt to upload an invalid document type
  Given the user is on the "document upload" page
  When the user selects the file "invalid_document.txt"
  And the user clicks the "upload" button
  Then the system should display the message "Invalid file type"
  And the document "invalid_document.txt" should not be uploaded

@smoke @regression
Scenario: Attempt to upload a document larger than the size limit
  Given the user is on the "document upload" page
  When the user selects the file "large_document.pdf"
  And the user clicks the "upload" button
  Then the system should display the message "File size exceeds limit"
  And the document "large_document.pdf" should not be uploaded

@smoke @regression
Scenario: Upload a document without selecting a file
  Given the user is on the "document upload" page
  When the user clicks the "upload" button
  Then the system should display the message "No file selected"
  And no document should be uploaded